from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


import jax
import jax.numpy as jnp

Array = jnp.ndarray

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "palatino"
plt.rcParams.update({
    "font.size": 12,         # default text size
    "axes.titlesize": 16,    # title
    "axes.labelsize": 14,    # x and y labels
    "xtick.labelsize": 12,   # x tick labels
    "ytick.labelsize": 12,   # y tick labels
    "legend.fontsize": 12,   # legend
    "font.weight": "bold",
})

# -----------------------------
# Pendulum black-box simulator
# -----------------------------
def pendulum_continuous_dynamics(x: Array, u: Array, params: dict) -> Array:
    """
    x = [phi, phi_dot], where phi = theta - pi (upright at 0).
    u = [tau]
    """
    phi, w = x
    tau = u[0]
    g = params.get("g", 9.81)
    m = params.get("m", 1.0)
    l = params.get("l", 0.5)
    b = params.get("b", 0.05)  # viscous damping

    # dynamics around upright (phi=0): -(g/l) * sin(phi)
    phi_dot = w
    w_dot = -(g / l) * jnp.sin(phi) - (b / (m * l * l)) * w + (1.0 / (m * l * l)) * tau
    return jnp.array([phi_dot, w_dot])

def rk4_step(f, x: Array, u: Array, dt: float, params: dict) -> Array:
    k1 = f(x, u, params)
    k2 = f(x + 0.5 * dt * k1, u, params)
    k3 = f(x + 0.5 * dt * k2, u, params)
    k4 = f(x + dt * k3, u, params)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def f_step(x: Array, u: Array, params: dict) -> Array:
    """Discrete-time step using RK4."""
    return rk4_step(pendulum_continuous_dynamics, x, u, params["dt"], params)

# -----------------------------
# Finite-difference linearization
# -----------------------------
def finite_diff_jacobians(f_step_fn, x: Array, u: Array, params: dict,
                          eps_x: float = 1e-5, eps_u: float = 1e-5) -> Tuple[Array, Array, Array]:
    """
    Returns A,B,c for x+ = A x + B u + c via central finite differences.
    """
    nx = x.shape[0]
    nu = u.shape[0]
    fxu = f_step_fn(x, u, params)

    # A = df/dx
    A_cols = []
    for i in range(nx):
        dx = jnp.zeros_like(x).at[i].set(eps_x)
        f_plus  = f_step_fn(x + dx, u, params)
        f_minus = f_step_fn(x - dx, u, params)
        A_cols.append((f_plus - f_minus) / (2.0 * eps_x))
    A = jnp.stack(A_cols, axis=1)  # [nx,nx]

    # B = df/du
    B_cols = []
    for j in range(nu):
        du = jnp.zeros_like(u).at[j].set(eps_u)
        f_plus  = f_step_fn(x, u + du, params)
        f_minus = f_step_fn(x, u - du, params)
        B_cols.append((f_plus - f_minus) / (2.0 * eps_u))
    B = jnp.stack(B_cols, axis=1)  # [nx,nu]

    # Affine residual
    c = fxu - A @ x - B @ u
    return A, B, c

# -----------------------------
# Costs
# -----------------------------
@dataclass
class ILQRCost:
    Q: Array       # [nx,nx]
    R: Array       # [nu,nu]
    Qf: Array      # [nx,nx]
    x_ref: Array   # [T+1,nx]
    u_ref: Array   # [T,nu]
    u_min: Optional[Array] = None  # [nu]
    u_max: Optional[Array] = None  # [nu]

@dataclass
class ILQROpts:
    max_iters: int = 60
    line_search: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)
    reg_init: float = 1e-6
    reg_mult_up: float = 10.0
    reg_mult_down: float = 0.3
    reg_max: float = 1e9
    clamp_controls: bool = True
    fd_eps_x: float = 1e-5
    fd_eps_u: float = 1e-5

def stage_cost(x: Array, u: Array, t: int, cost: ILQRCost) -> float:
    dx = x - cost.x_ref[t]
    du = u - cost.u_ref[t]
    return dx @ cost.Q @ dx + du @ cost.R @ du

def term_cost(xT: Array, cost: ILQRCost) -> float:
    dx = xT - cost.x_ref[-1]
    return dx @ cost.Qf @ dx

def clamp_u(u: Array, cost: ILQRCost) -> Array:
    if (cost.u_min is None) or (cost.u_max is None):
        return u
    return jnp.clip(u, cost.u_min, cost.u_max)

# -----------------------------
# Rollout & cost
# -----------------------------
def rollout(f_step_fn, x0: Array, U: Array, params: dict) -> Array:
    def body(x, u):
        xn = f_step_fn(x, u, params)
        return xn, xn
    _, Xs = jax.lax.scan(body, x0, U)
    return jnp.vstack([x0[None, :], Xs])

def traj_cost(X: Array, U: Array, cost: ILQRCost) -> float:
    T = U.shape[0]
    stage = jnp.sum(jax.vmap(lambda t: stage_cost(X[t], U[t], t, cost))(jnp.arange(T)))
    return stage + term_cost(X[-1], cost)

# -----------------------------
# Quadratic expansions (analytic for LQR tracking)
# -----------------------------
def quad_stage_terms(x: Array, u: Array, t: int, cost: ILQRCost):
    # For quadratic tracking, these are constant:
    dx = x - cost.x_ref[t]
    du = u - cost.u_ref[t]
    l = dx @ cost.Q @ dx + du @ cost.R @ du
    lx = 2.0 * cost.Q @ dx
    lu = 2.0 * cost.R @ du
    lxx = 2.0 * cost.Q
    luu = 2.0 * cost.R
    lux = jnp.zeros((u.shape[0], x.shape[0]))
    return l, lx, lu, lxx, luu, lux

def quad_term(xT: Array, cost: ILQRCost):
    dx = xT - cost.x_ref[-1]
    V = dx @ cost.Qf @ dx
    Vx = 2.0 * cost.Qf @ dx
    Vxx = 2.0 * cost.Qf
    return V, Vx, Vxx

# -----------------------------
# Backward pass
# -----------------------------
def backward_pass(As, Bs, X, U, cost: ILQRCost, reg: float):
    T = U.shape[0]
    nx = X.shape[1]
    nu = U.shape[1]

    V, Vx, Vxx = quad_term(X[-1], cost)
    Ks = []
    ks = []

    dV = 0.0

    for t in range(T - 1, -1, -1):
        _, lx, lu, lxx, luu, lux = quad_stage_terms(X[t], U[t], t, cost)

        A = As[t]; B = Bs[t]
        Vxx_reg = Vxx + reg * jnp.eye(nx)

        Qx  = lx + A.T @ Vx
        Qu  = lu + B.T @ Vx
        Qxx = lxx + A.T @ Vxx_reg @ A
        Quu = luu + B.T @ Vxx_reg @ B
        Qux = lux + B.T @ Vxx_reg @ A

        # Stabilize Quu (Cholesky)
        L = jnp.linalg.cholesky(Quu)
        # Solve Quu * y = rhs
        def cho_solve(L, rhs):
            y = jax.scipy.linalg.cho_solve((L, False), rhs)
            return y

        k = -cho_solve(L, Qu)
        K = -cho_solve(L, Qux)

        Vx = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
        Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
        Vxx = 0.5 * (Vxx + Vxx.T)  # symmetrize

        dV = dV + (-0.5 * (k @ Quu @ k) - (k @ Qu))

        ks.append(k)
        Ks.append(K)

    ks = ks[::-1]
    Ks = Ks[::-1]
    return jnp.stack(Ks, 0), jnp.stack(ks, 0), dV

# -----------------------------
# Forward pass with line search
# -----------------------------
def forward_pass(f_step_fn, x0, U, K, k, X_old, params, cost: ILQRCost, alpha: float):
    def body(carry, t):
        x = carry
        u_ff = U[t] + alpha * k[t] + K[t] @ (x - X_old[t])
        u_new = clamp_u(u_ff, cost)
        x_new = f_step_fn(x, u_new, params)
        return x_new, (x_new, u_new)
    xT, (X_body, U_body) = jax.lax.scan(body, x0, jnp.arange(U.shape[0]))
    X_new = jnp.vstack([x0[None, :], X_body])
    return X_new, U_body

# -----------------------------
# iLQR main
# -----------------------------
@dataclass
class ILQRResult:
    X: Array
    U: Array
    cost: float
    iters: int
    reg: float

def ilqr_pendulum_fd(x0: Array,
                     U_init: Array,
                     params: dict,
                     cost: ILQRCost,
                     opts: ILQROpts = ILQROpts()) -> ILQRResult:
    T, nu = U_init.shape
    nx = x0.shape[0]

    U = U_init
    #X = rollout(f_step, x0, U, params)
    X = jnp.ones((T+1, nx))
    J = traj_cost(X, U, cost)

    reg = opts.reg_init

    for it in range(opts.max_iters):
        # Finite-diff linearization along (X,U)
        def lin_one(t):
            A, B, c = finite_diff_jacobians(f_step, X[t], U[t], params,
                                            eps_x=opts.fd_eps_x, eps_u=opts.fd_eps_u)
            return A, B, c
        As, Bs, Cs = jax.vmap(lin_one)(jnp.arange(T))

        # Backward pass
        K, k, dV = backward_pass(As, Bs, X, U, cost, reg)

        # Line search
        improved = False
        best = (X, U, J)
        for a in opts.line_search:
            X_new, U_new = forward_pass(f_step, x0, U, K, k, X, params, cost, a)
            J_new = traj_cost(X_new, U_new, cost)
            if J_new < J:
                improved = True
                best = (X_new, U_new, J_new)
                break

        if improved:
            X, U, J = best
            reg = jnp.maximum(opts.reg_init, reg * opts.reg_mult_down)
        else:
            reg = reg * opts.reg_mult_up
            if reg > opts.reg_max:
                break

    return ILQRResult(X=X, U=U, cost=J, iters=it+1, reg=reg)

# -----------------------------
# Demo / Example
# -----------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    # System params
    params = {
        "g": 9.81,
        "m": 1.0,
        "l": 0.5,
        "b": 0.05,
        "dt": 0.02
    }

    nx, nu, T = 2, 1, 200

    # "Diffused" reference (mock): near upright zero, small noise
    x_ref = 0.05 * jax.random.normal(key, (T+1, nx))
    u_ref = 0.05 * jax.random.normal(key, (T,   nu))

    # Initial control guess: reference controls
    U0 = u_ref.copy()

    # Costs: emphasize angle stabilization
    Q  = jnp.diag(jnp.array([10.0, 1.0], dtype=jnp.float32))
    R  = jnp.diag(jnp.array([1e-2], dtype=jnp.float32))
    Qf = jnp.diag(jnp.array([50.0, 5.0], dtype=jnp.float32))

    cost = ILQRCost(Q=Q, R=R, Qf=Qf, x_ref=x_ref, u_ref=u_ref,
                    u_min=jnp.array([-3.0]), u_max=jnp.array([3.0]))
    opts = ILQROpts(max_iters=60)

    # Start state (can be the ref's first state or something specific)
    x0 = jnp.array([jnp.pi, 0.0])  # 0.2 rad from upright, zero speed

    result = ilqr_pendulum_fd(x0, U0, params, cost, opts)
    print(f"Done in {int(result.iters)} iters, final cost {float(result.cost):.4f}, reg={float(result.reg):.3e}")

    #(Optional) quick check that it stabilizes (no plotting here):
    print("First/last states:", result.X[0], result.X[-1])

    # ----------------------------------------------------------
    # After running iLQR:
    # ----------------------------------------------------------
    t = np.arange(result.X.shape[0]) * params["dt"]
    phi = np.array(result.X[:, 0])        # angle deviation (rad)
    omega = np.array(result.X[:, 1])      # angular velocity (rad/s)
    u = np.array(result.U[:, 0])          # torque (Nm)

    # ------------------ Static plots --------------------------
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axs[0].plot(t, phi, label=r"Angle deviation $\phi$")
    axs[0].set_ylabel(r"$\phi\,[\mathrm{rad}]$")

    axs[1].plot(t, omega, label=r"Angular velocity $\omega$", color="orange")
    axs[1].set_ylabel(r"$\omega\,[\mathrm{rad/s}]$")

    axs[2].plot(t[:-1], u, label=r"Control torque $u$", color="green")
    axs[2].set_ylabel(r"$u\,[\mathrm{N\,m}]$")
    axs[2].set_xlabel(r"$t\,[\mathrm{s}]$")

    for ax in axs:
        ax.grid(True)
        ax.legend()
    plt.suptitle("iLQR-Feasible Pendulum Trajectory")
    plt.tight_layout()
    plt.show()

    # ------------------ Animation -----------------------------
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.set_xlim(-0.7, 0.7)
    ax2.set_ylim(-0.7, 0.7)
    ax2.set_aspect('equal')
    ax2.set_title("Pendulum Animation (upright stabilization)")

    line, = ax2.plot([], [], 'o-', lw=3, color='royalblue')
    trace, = ax2.plot([], [], 'r--', lw=1, alpha=0.4)

    l = params["l"]
    x_trace, y_trace = [], []

    # def update(frame):
    #     phi_f = phi[frame]
    #     # pivot at (0,0), upright at φ=0 → rod points upward
    #     x_tip = l * np.sin(phi_f)
    #     y_tip = l * np.cos(phi_f)
    #     line.set_data([0, x_tip], [0, y_tip])
    #     x_trace.append(x_tip)
    #     y_trace.append(y_tip)
    #     trace.set_data(x_trace, y_trace)
    #     return line, trace


    trace_len = 20  # number of past frames to keep in trace
    x_trace, y_trace = [], []


    def update(frame):
        phi_f = phi[frame]
        # pendulum tip coordinates
        x_tip = l * np.sin(phi_f)
        y_tip = l * np.cos(phi_f)

        # update pendulum rod
        line.set_data([0, x_tip], [0, y_tip])

        # update trace buffer (keep only recent N points)
        x_trace.append(x_tip)
        y_trace.append(y_tip)
        if len(x_trace) > trace_len:
            x_trace.pop(0)
            y_trace.pop(0)

        # fading trace using color alpha gradient
        trace.set_data(x_trace, y_trace)
        trace.set_alpha(0.3 + 0.7 * (len(x_trace) / trace_len))  # subtle fade

        return line, trace

    ani = animation.FuncAnimation(
        fig2, update, frames=len(phi), interval=1000*params["dt"], blit=True
    )
    plt.show()
