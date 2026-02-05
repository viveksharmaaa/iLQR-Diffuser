# ilqr_cartpole_fd.py
#https://stanfordasl.github.io/aa203/sp2223/pdfs/homework/hw2.pdf Cartpole
#https://github.com/harwiltz/ilqr/blob/master/ilqr_cartpole.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

# ==========================================================
# 1) CARTPOLE DYNAMICS (upright at theta=0) + RK4 STEP
#    State x = [x, xdot, theta, thetadot], control u = [force]
#    Parameters: M (cart mass), m (pole mass), l (half-length),
#                b (cart viscous), g, dt
#    This is a standard continuous-time model, integrated with RK4.
# ==========================================================
def cartpole_continuous_dynamics(x: Array, u: Array, params: dict) -> Array:
    X, Xd, th, thd = x
    F = u[0]
    M = params.get("M", 1.0)
    m = params.get("m", 0.1)
    l = params.get("l", 0.5)   # half-length (center of mass at l)
    b = params.get("b", 0.0)
    g = params.get("g", 9.81)

    s = jnp.sin(th)
    c = jnp.cos(th)

    # Common denominator (to avoid singularities near horizontal)
    denom = M + m - m * c * c

    # Accelerations (upright at th=0)
    Xdd = (F - b * Xd + m * l * thd * thd * s - m * g * l * s * c) / denom
    thdd = (g * s - c * (F - b * Xd + m * l * thd * thd * s) / (M + m)) / (l * (4.0 / 3.0 - (m * c * c) / (M + m)))

    return jnp.array([Xd, Xdd, th, thdd])

def rk4_step(f, x: Array, u: Array, dt: float, params: dict) -> Array:
    k1 = f(x, u, params)
    k2 = f(x + 0.5 * dt * k1, u, params)
    k3 = f(x + 0.5 * dt * k2, u, params)
    k4 = f(x + dt * k3, u, params)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def f_step(x: Array, u: Array, params: dict) -> Array:
    return rk4_step(cartpole_continuous_dynamics, x, u, params["dt"], params)

# ==========================================================
# 2) FINITE-DIFFERENCE LINEARIZATION
# ==========================================================
def finite_diff_jacobians(f_step_fn, x: Array, u: Array, params: dict,
                          eps_x: float = 1e-5, eps_u: float = 1e-5) -> Tuple[Array, Array, Array]:
    nx = x.shape[0]; nu = u.shape[0]
    fxu = f_step_fn(x, u, params)
    # df/dx
    A_cols = []
    for i in range(nx):
        dx = jnp.zeros_like(x).at[i].set(eps_x)
        f_plus  = f_step_fn(x + dx, u, params)
        f_minus = f_step_fn(x - dx, u, params)
        A_cols.append((f_plus - f_minus) / (2.0 * eps_x))
    A = jnp.stack(A_cols, axis=1)
    # df/du
    B_cols = []
    for j in range(nu):
        du = jnp.zeros_like(u).at[j].set(eps_u)
        f_plus  = f_step_fn(x, u + du, params)
        f_minus = f_step_fn(x, u - du, params)
        B_cols.append((f_plus - f_minus) / (2.0 * eps_u))
    B = jnp.stack(B_cols, axis=1)
    # affine residual
    c = fxu - A @ x - B @ u
    return A, B, c

# ==========================================================
# 3) COSTS (quadratic tracking of diffused ref)
# ==========================================================
@dataclass
class ILQRCost:
    Q: Array
    R: Array
    Qf: Array
    x_ref: Array   # [T+1, nx]
    u_ref: Array   # [T,   nu]
    u_min: Optional[Array] = None
    u_max: Optional[Array] = None

@dataclass
class ILQROpts:
    max_iters: int = 60
    line_search: tuple = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)
    reg_init: float = 1e-6
    reg_mult_up: float = 10.0
    reg_mult_down: float = 0.3
    reg_max: float = 1e9
    fd_eps_x: float = 1e-5
    fd_eps_u: float = 1e-5

def clamp_u(u: Array, cost: ILQRCost) -> Array:
    if (cost.u_min is None) or (cost.u_max is None):
        return u
    return jnp.clip(u, cost.u_min, cost.u_max)

def rollout(f_step_fn, x0: Array, U: Array, params: dict) -> Array:
    def body(x, u):
        xn = f_step_fn(x, u, params)
        return xn, xn
    _, Xs = jax.lax.scan(body, x0, U)
    return jnp.vstack([x0[None, :], Xs])

def stage_cost(x: Array, u: Array, t: int, cost: ILQRCost) -> float:
    dx = x - cost.x_ref[t]
    du = u - cost.u_ref[t]
    return dx @ cost.Q @ dx + du @ cost.R @ du

def term_cost(xT: Array, cost: ILQRCost) -> float:
    dx = xT - cost.x_ref[-1]
    return dx @ cost.Qf @ dx

def traj_cost(X: Array, U: Array, cost: ILQRCost) -> float:
    T = U.shape[0]
    stage = jnp.sum(jax.vmap(lambda t: stage_cost(X[t], U[t], t, cost))(jnp.arange(T)))
    return stage + term_cost(X[-1], cost)

# Quadratic expansions (analytic for quadratic tracking)
def quad_stage_terms(x: Array, u: Array, t: int, cost: ILQRCost):
    dx = x - cost.x_ref[t]; du = u - cost.u_ref[t]
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

# ==========================================================
# 4) iLQR (backward + forward + line search)
# ==========================================================
def backward_pass(As, Bs, X, U, cost: ILQRCost, reg: float):
    T = U.shape[0]; nx = X.shape[1]
    V, Vx, Vxx = quad_term(X[-1], cost)
    Ks = []; ks = []
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
        L = jnp.linalg.cholesky(Quu)
        def cho_solve(L, rhs):  # stable solve
            return jax.scipy.linalg.cho_solve((L, False), rhs)
        k = -cho_solve(L, Qu)
        K = -cho_solve(L, Qux)
        Vx  = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
        Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
        Vxx = 0.5 * (Vxx + Vxx.T)
        dV += (-0.5 * (k @ Quu @ k) - (k @ Qu))
        ks.append(k); Ks.append(K)
    ks = ks[::-1]; Ks = Ks[::-1]
    return jnp.stack(Ks, 0), jnp.stack(ks, 0), dV

def forward_pass(f_step_fn, x0, U, K, k, X_old, params, cost: ILQRCost, alpha: float):
    def body(x, t):
        u_ff = U[t] + alpha * k[t] + K[t] @ (x - X_old[t])
        u_new = clamp_u(u_ff, cost)
        x_new = f_step_fn(x, u_new, params)
        return x_new, (x_new, u_new)
    xT, (X_body, U_body) = jax.lax.scan(body, x0, jnp.arange(U.shape[0]))
    X_new = jnp.vstack([x0[None, :], X_body])
    return X_new, U_body

@dataclass
class ILQROptsFull(ILQROpts):
    pass

@dataclass
class ILQRResult:
    X: Array
    U: Array
    cost: float
    iters: int
    reg: float

def ilqr_cartpole_fd(x0: Array, U_init: Array, params: dict,
                     cost: ILQRCost, opts: ILQROpts = ILQROpts()) -> ILQRResult:
    U = U_init
    X = rollout(f_step, x0, U, params)
    J = traj_cost(X, U, cost)
    reg = opts.reg_init

    for it in range(opts.max_iters):
        # Finite-diff linearization on current (X,U)
        def lin_one(t):
            A, B, c = finite_diff_jacobians(
                f_step, X[t], U[t], params, eps_x=opts.fd_eps_x, eps_u=opts.fd_eps_u
            )
            return A, B, c
        As, Bs, Cs = jax.vmap(lin_one)(jnp.arange(U.shape[0]))

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

    return ILQRResult(X=X, U=U, cost=J, iters=it + 1, reg=reg)

# ==========================================================
# 5) DEMO + PLOTS + ANIMATION
# ==========================================================
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    # System parameters
    params = dict(M=1.0, m=0.1, l=0.5, b=0.02, g=9.81, dt=0.02)

    nx, nu, T = 4, 1, 300

    # Mock "diffused" reference near upright (theta≈0)
    x_ref = 0.02 * jax.random.normal(key, (T + 1, nx))
    u_ref = 0.05 * jax.random.normal(key, (T, nu))
    #u_ref = 0.5 * jnp.sin(jnp.linspace(0, 4 * jnp.pi, T)).reshape(-1, 1)

    # Initial control guess
    U0 = u_ref.copy()

    # Costs: strong on theta (upright), moderate on cart position & velocities
    Q = jnp.diag(jnp.array([1.0, 0.1, 20.0, 1.5], dtype=jnp.float32))
    R = jnp.diag(jnp.array([1e-2], dtype=jnp.float32))
    Qf = jnp.diag(jnp.array([5.0, 0.2, 80.0, 5.0], dtype=jnp.float32))

    # Control limits
    umax = 20.0
    cost = ILQRCost(Q=Q, R=R, Qf=Qf, x_ref=x_ref, u_ref=u_ref,
                    u_min=jnp.array([-umax]), u_max=jnp.array([umax]))

    # Start state: slightly off upright
    x0 = jnp.array([0.0, 0.0, 0.01, 0.0])

    # Run iLQR
    result = ilqr_cartpole_fd(x0, U0, params, cost, ILQROpts(max_iters=80))
    print(f"iLQR done in {int(result.iters)} iters, cost={float(result.cost):.4f}, reg={float(result.reg):.3e}")

    # ---------- Static plots ----------
    t = np.arange(result.X.shape[0]) * params["dt"]
    X = np.array(result.X)
    U = np.array(result.U[:, 0])

    x = X[:, 0]
    xdot = X[:, 1]
    th = X[:, 2]
    thd = X[:, 3]

    fig, axs = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    axs[0].plot(t, x);    axs[0].set_ylabel("cart x [m]")
    axs[1].plot(t, th);   axs[1].set_ylabel("theta [rad] (upright=0)")
    axs[2].plot(t, xdot); axs[2].set_ylabel("xdot [m/s]")
    axs[3].plot(t[:-1], U); axs[3].set_ylabel("u [N]"); axs[3].set_xlabel("time [s]")
    for ax in axs: ax.grid(True)
    plt.suptitle("Cartpole iLQR (finite-diff dynamics) — tracking diffused reference")
    plt.tight_layout()
    plt.show()

    # ---------- Animation ----------
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.set_xlim(-2.2, 2.2)
    ax2.set_ylim(-0.6, 1.2)
    ax2.set_aspect('equal')
    ax2.set_title("Cartpole Animation (upright stabilization)")
    ax2.axhline(0, color='k', lw=1)

    # Cart as a rectangle, pole as a line
    cart_w, cart_h = 0.3, 0.2
    l = params["l"]  # half-length to COM; pole tip approx at 2*l
    cart, = ax2.plot([], [], 'k-', lw=3)
    pole, = ax2.plot([], [], 'b-', lw=3)
    tip_trace, = ax2.plot([], [], 'r--', alpha=0.5)

    trace_x, trace_y = [], []

    def update(k):
        xc = x[k]
        thk = th[k]
        # cart rectangle corners
        xL = xc - cart_w/2; xR = xc + cart_w/2
        yB = 0.0;          yT = cart_h
        cart.set_data([xL, xR, xR, xL, xL], [yB, yB, yT, yT, yB])

        # Pole pivot at (xc, yT). Tip at:
        tip_x = xc + (2*l) * np.sin(thk)
        tip_y = yT + (2*l) * np.cos(thk)
        pole.set_data([xc, tip_x], [yT, tip_y])

        trace_x.append(tip_x); trace_y.append(tip_y)
        tip_trace.set_data(trace_x, trace_y)
        return cart, pole, tip_trace

    ani = animation.FuncAnimation(fig2, update, frames=len(t),
                                  interval=1000*params["dt"], blit=True)
    plt.show()

    # To save:
    # ani.save("cartpole_ilqr.mp4", fps=int(1/params["dt"]), dpi=150, codec="libx264")
