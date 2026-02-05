from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax
import jax.numpy as jnp
Array = jnp.ndarray
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "palatino",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})
# ================================================================
# Acrobot dynamics (continuous-time)
# ================================================================
def acrobot_continuous_dynamics(x: Array, u: Array, p: dict) -> Array:
    """
    State: x = [θ1, θ2, θ1_dot, θ2_dot]
    Control: u = [τ2]  (torque at second joint only)
    """
    th1, th2, dth1, dth2 = x
    tau = u[0]
    m1, m2 = p.get("m1", 1.0), p.get("m2", 1.0)
    l1, l2 = p.get("l1", 0.5), p.get("l2", 0.5)
    lc1, lc2 = p.get("lc1", l1 / 2.0), p.get("lc2", l2 / 2.0)
    I1, I2 = p.get("I1", m1 * l1**2 / 3.0), p.get("I2", m2 * l2**2 / 3.0)
    g, b1, b2 = p.get("g", 9.81), p.get("b1", 0.05), p.get("b2", 0.05)
    c2 = jnp.cos(th2)
    s2 = jnp.sin(th2)
    # Inertia matrix
    d11 = I1 + I2 + m2 * l1**2 + 2 * m2 * l1 * lc2 * c2
    d12 = I2 + m2 * l1 * lc2 * c2
    d21 = d12
    d22 = I2
    M = jnp.array([[d11, d12],
                   [d21, d22]])
    # Coriolis/Centrifugal
    h = -m2 * l1 * lc2 * s2
    C = jnp.array([[h * dth2, h * (dth1 + dth2)],
                   [-h * dth1, 0.0]])
    # Gravity
    g1 = (m1 * lc1 + m2 * l1) * g * jnp.sin(th1) + m2 * lc2 * g * jnp.sin(th1 + th2)
    g2 = m2 * lc2 * g * jnp.sin(th1 + th2)
    G = jnp.array([g1, g2])
    # Damping
    B = jnp.array([[b1, 0.0],
                   [0.0, b2]])
    # Control (only at joint 2)
    tau_vec = jnp.array([0.0, tau])
    # Compute accelerations
    rhs = tau_vec - C @ jnp.array([dth1, dth2]) - G - B @ jnp.array([dth1, dth2])
    ddth = jnp.linalg.solve(M, rhs)
    return jnp.array([dth1, dth2, ddth[0], ddth[1]])
# ================================================================
# Discretization (RK4)
# ================================================================
def rk4_step(f, x, u, dt, params):
    k1 = f(x, u, params)
    k2 = f(x + 0.5 * dt * k1, u, params)
    k3 = f(x + 0.5 * dt * k2, u, params)
    k4 = f(x + dt * k3, u, params)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
def f_step(x, u, params):
    return rk4_step(acrobot_continuous_dynamics, x, u, params["dt"], params)
# ================================================================
# Finite-difference linearization
# ================================================================
def finite_diff_jacobians(f_step_fn, x, u, params, eps_x=1e-5, eps_u=1e-5):
    nx, nu = x.shape[0], u.shape[0]
    fxu = f_step_fn(x, u, params)
    # A = df/dx
    A_cols = []
    for i in range(nx):
        dx = jnp.zeros_like(x).at[i].set(eps_x)
        f_plus  = f_step_fn(x + dx, u, params)
        f_minus = f_step_fn(x - dx, u, params)
        A_cols.append((f_plus - f_minus) / (2*eps_x))
    A = jnp.stack(A_cols, axis=1)
    # B = df/du
    B_cols = []
    for j in range(nu):
        du = jnp.zeros_like(u).at[j].set(eps_u)
        f_plus  = f_step_fn(x, u + du, params)
        f_minus = f_step_fn(x, u - du, params)
        B_cols.append((f_plus - f_minus) / (2*eps_u))
    B = jnp.stack(B_cols, axis=1)
    c = fxu - A @ x - B @ u
    return A, B, c
# ================================================================
# Cost structures
# ================================================================
@dataclass
class ILQRCost:
    Q: Array
    R: Array
    Qf: Array
    x_ref: Array
    u_ref: Array
    u_min: Optional[Array] = None
    u_max: Optional[Array] = None
@dataclass
class ILQROpts:
    max_iters: int = 60
    line_search: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05)
    reg_init: float = 1e-6
    fd_eps_x: float = 1e-5
    fd_eps_u: float = 1e-5
# ================================================================
# Stage & terminal cost
# ================================================================
def stage_cost(x, u, t, cost: ILQRCost):
    dx = x - cost.x_ref[t]
    du = u - cost.u_ref[t]
    return dx @ cost.Q @ dx + du @ cost.R @ du
def term_cost(xT, cost: ILQRCost):
    dx = xT - cost.x_ref[-1]
    return dx @ cost.Qf @ dx
def clamp_u(u, cost: ILQRCost):
    if cost.u_min is None: return u
    return jnp.clip(u, cost.u_min, cost.u_max)
# ================================================================
# Rollout + trajectory cost
# ================================================================
def rollout(f_step_fn, x0, U, params):
    def step_fn(x, u):
        xn = f_step_fn(x, u, params)
        return xn, xn
    _, X_body = jax.lax.scan(step_fn, x0, U)
    return jnp.vstack([x0[None, :], X_body])
def traj_cost(X, U, cost: ILQRCost):
    T = U.shape[0]
    stage = jnp.sum(jax.vmap(lambda t: stage_cost(X[t], U[t], t, cost))(jnp.arange(T)))
    return stage + term_cost(X[-1], cost)
# ================================================================
# Quadratic expansions
# ================================================================
def quad_stage_terms(x, u, t, cost: ILQRCost):
    dx = x - cost.x_ref[t]
    du = u - cost.u_ref[t]
    l = dx @ cost.Q @ dx + du @ cost.R @ du
    lx = 2 * cost.Q @ dx
    lu = 2 * cost.R @ du
    lxx = 2 * cost.Q
    luu = 2 * cost.R
    lux = jnp.zeros((u.shape[0], x.shape[0]))
    return l, lx, lu, lxx, luu, lux
def quad_term(xT, cost: ILQRCost):
    dx = xT - cost.x_ref[-1]
    V = dx @ cost.Qf @ dx
    Vx = 2 * cost.Qf @ dx
    Vxx = 2 * cost.Qf
    return V, Vx, Vxx
# ================================================================
# Backward pass
# ================================================================
def backward_pass(As, Bs, X, U, cost: ILQRCost, reg: float):
    T = U.shape[0]
    nx, nu = X.shape[1], U.shape[1]
    V, Vx, Vxx = quad_term(X[-1], cost)
    Ks, ks = [], []
    dV = 0.0
    for t in range(T - 1, -1, -1):
        _, lx, lu, lxx, luu, lux = quad_stage_terms(X[t], U[t], t, cost)
        A, B = As[t], Bs[t]
        Vxx_reg = Vxx + reg * jnp.eye(nx)
        Qx = lx + A.T @ Vx
        Qu = lu + B.T @ Vx
        Qxx = lxx + A.T @ Vxx_reg @ A
        Quu = luu + B.T @ Vxx_reg @ B
        Qux = lux + B.T @ Vxx_reg @ A
        L = jnp.linalg.cholesky(Quu)
        k = -jax.scipy.linalg.cho_solve((L, False), Qu)
        K = -jax.scipy.linalg.cho_solve((L, False), Qux)
        Vx = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
        Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
        Vxx = 0.5 * (Vxx + Vxx.T)
        dV += -0.5 * (k @ Quu @ k) - (k @ Qu)
        ks.append(k)
        Ks.append(K)
    ks, Ks = ks[::-1], Ks[::-1]
    return jnp.stack(Ks), jnp.stack(ks), dV
# ================================================================
# Forward pass with line search
# ================================================================
def forward_pass(f_step_fn, x0, U, K, k, X_old, params, cost: ILQRCost, alpha: float):
    def step_fn(x, t):
        u_ff = U[t] + alpha * k[t] + K[t] @ (x - X_old[t])
        u_new = clamp_u(u_ff, cost)
        x_new = f_step_fn(x, u_new, params)
        return x_new, (x_new, u_new)
    _, (X_body, U_body) = jax.lax.scan(step_fn, x0, jnp.arange(U.shape[0]))
    return jnp.vstack([x0[None, :], X_body]), U_body
# ================================================================
# iLQR main
# ================================================================
@dataclass
class ILQRResult:
    X: Array
    U: Array
    cost: float
    iters: int
def ilqr_acrobot_fd(x0, U_init, params, cost: ILQRCost, opts: ILQROpts) -> ILQRResult:
    U, X = U_init, rollout(f_step, x0, U_init, params)
    J = traj_cost(X, U, cost)
    reg = opts.reg_init
    for it in range(opts.max_iters):
        # linearize
        def lin_one(t):
            return finite_diff_jacobians(f_step, X[t], U[t], params, opts.fd_eps_x, opts.fd_eps_u)
        As, Bs, _ = jax.vmap(lin_one)(jnp.arange(U.shape[0]))
        K, k, dV = backward_pass(As, Bs, X, U, cost, reg)
        improved = False
        for a in opts.line_search:
            Xn, Un = forward_pass(f_step, x0, U, K, k, X, params, cost, a)
            Jn = traj_cost(Xn, Un, cost)
            if Jn < J:
                X, U, J = Xn, Un, Jn
                improved = True
                break
        if not improved:
            reg *= 10
        else:
            reg = max(reg * 0.5, 1e-6)
    return ILQRResult(X, U, J, it + 1)
# ================================================================
# Demo
# ================================================================

# ---------- Better initial guess for Acrobot (swing-up + stabilize) ----------
def _wrap_angle(a):
    return (a + jnp.pi) % (2*jnp.pi) - jnp.pi

def _acrobot_energy(x, p):
    th1, th2, d1, d2 = x
    m1, m2 = p["m1"], p["m2"]
    l1, lc1, lc2 = p["l1"], p["lc1"], p["lc2"]
    I1, I2 = p["I1"], p["I2"]
    g = p["g"]

    c2 = jnp.cos(th2)
    # inertia
    d11 = I1 + I2 + m2*l1*l1 + 2*m2*l1*lc2*c2
    d12 = I2 + m2*l1*lc2*c2
    d21 = d12
    d22 = I2
    M = jnp.array([[d11, d12],
                   [d21, d22]])
    dq = jnp.array([d1, d2])
    T = 0.5 * dq @ (M @ dq)

    # potential (zero at downward θ1=0, θ2=0 if we subtract V_down)
    y1 = -lc1*jnp.cos(th1)
    y2 = -l1*jnp.cos(th1) - lc2*jnp.cos(th1+th2)
    V = m1*g*y1 + m2*g*y2
    V_down = -(m1*lc1 + m2*(l1+lc2)) * g
    return T + (V - V_down)

def _acrobot_warm_torque(x, p, gains, umin, umax):
    """
    Energy pump when far from upright; PD on elbow near upright.
    Upright target: θ1≈π, θ2≈0 (both links up), zero velocities.
    """
    th1, th2, d1, d2 = x
    # energy target (raise both COMs): ΔV = 2*g*(m1*lc1 + m2*(l1+lc2))
    E_star = 2.0 * p["g"] * (p["m1"]*p["lc1"] + p["m2"]*(p["l1"] + p["lc2"]))
    E = _acrobot_energy(x, p)

    # swing-up torque (only joint 2 actuated)
    s = jnp.sign((d1 + d2) * jnp.cos(th1 + th2))
    tau_pump = -gains["kE"] * (E - E_star) * s - gains["kd_pump"] * d2

    # near-upright detection
    e1 = _wrap_angle(th1 - jnp.pi)
    e2 = _wrap_angle(th2 - 0.0)
    near = (jnp.abs(e1) < gains["pos_tol"]) & (jnp.abs(e2) < gains["pos_tol"]) \
           & (jnp.abs(d1) < gains["vel_tol"]) & (jnp.abs(d2) < gains["vel_tol"])

    # small-angle stabilizer on elbow (could also use sum angle)
    tau_stab = -gains["kp_upr"] * e2 - gains["kd_upr"] * d2

    tau = jnp.where(near, tau_stab, tau_pump)
    return jnp.clip(tau, umin, umax)

def make_acrobot_initial_guess(x0, T, params,
                               u_limits=(-6.0, 6.0),
                               gains=None):
    """
    Returns (x_ref, u_ref) with shapes (T+1,4) and (T,1).
    Uses your existing RK4 discretization f_step(...) for accuracy.
    """
    if gains is None:
        gains = dict(
            kE=4.0,        # energy pump gain
            kd_pump=0.4,   # damping on dθ2 during pumping
            kp_upr=18.0,   # elbow PD near upright
            kd_upr=2.5,
            pos_tol=0.35,  # ~20 degrees
            vel_tol=1.2
        )
    umin, umax = u_limits

    def scan_step(x, _):
        tau = _acrobot_warm_torque(x, params, gains, umin, umax)
        u = jnp.array([tau])                 # (1,)
        x_next = f_step(x, u, params)        # uses your RK4
        return x_next, (x_next, u)

    _, (X_body, U_body) = jax.lax.scan(scan_step, x0, jnp.arange(T))
    x_ref = jnp.vstack([x0[None, :], X_body])     # (T+1,4)
    u_ref = U_body                                 # (T,1)
    return x_ref, u_ref

if __name__ == "__main__":
    params = dict(m1=1.0, m2=1.0, l1=0.5, l2=0.5, lc1=0.25, lc2=0.25,
                  I1=0.02, I2=0.02, b1=0.05, b2=0.05, g=9.81, dt=0.02)
    nx, nu, T = 4, 1, 250
    key = jax.random.PRNGKey(0)
    # x_ref = jnp.zeros((T+1, nx))
    # u_ref = jnp.zeros((T, nu))
    x0 = jnp.array([jnp.pi, 0.0, 0.0, 0.0])
    # Before calling iLQR:
    x_ref, u_ref = make_acrobot_initial_guess(x0, T, params)


    Q = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0]))
    R = jnp.diag(jnp.array([0.01]))
    Qf = 10 * Q
    cost = ILQRCost(Q, R, Qf, x_ref, u_ref,u_min=jnp.array([-5.0]), u_max=jnp.array([5.0]))
    opts = ILQROpts(max_iters=50)
    result = ilqr_acrobot_fd(x0, u_ref.copy(), params, cost, opts)
    print(f"Done in {result.iters} iters, final cost={float(result.cost):.3f}")
    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    t = np.arange(result.X.shape[0]) * params["dt"]
    theta_1, theta_2 = np.array(result.X[:,0]), np.array(result.X[:,1])
    omega_1, omega_2 = np.array(result.X[:,2]), np.array(result.X[:,3])
    u = np.array(result.U[:,0])
    fig, axs = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    axs[0].plot(t, theta_1, label=r"$\theta_1$")
    axs[0].plot(t, theta_2, label=r"$\theta_2$")
    axs[0].set_ylabel("angles [rad]"); axs[0].legend(); axs[0].grid(True)
    axs[1].plot(t, omega_1, label=r"$\dot{\theta}_1$")
    axs[1].plot(t, omega_2, label=r"$\dot{\theta}_2$")
    axs[1].set_ylabel("rates [rad/s]"); axs[1].legend(); axs[1].grid(True)
    axs[2].plot(t[:-1], u, color="green")
    axs[2].set_ylabel("u [Nm]"); axs[2].grid(True)
    axs[3].plot(t, theta_1 + theta_2, color="purple", label="link sum")
    axs[3].set_ylabel(r"$\theta_1+\theta_2$")
    axs[3].set_xlabel("time [s]"); axs[3].legend(); axs[3].grid(True)
    plt.tight_layout()
    plt.show()
    # ------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.set_title("Acrobot iLQR Stabilization")
    line, = ax2.plot([], [], 'o-', lw=3, color='royalblue')
    l1, l2 = params["l1"], params["l2"]
    def update(frame):
        t1, t2 = theta_1[frame], theta_2[frame]
        x1 = l1 * np.sin(t1)
        y1 = -l1 * np.cos(t1)
        x2 = x1 + l2 * np.sin(t1 + t2)
        y2 = y1 - l2 * np.cos(t1 + t2)
        line.set_data([0, x1, x2], [0, y1, y2])
        return line,
    ani = animation.FuncAnimation(fig2, update, frames=len(theta_1),
                                  interval=1000*params["dt"], blit=True)
    plt.show()
