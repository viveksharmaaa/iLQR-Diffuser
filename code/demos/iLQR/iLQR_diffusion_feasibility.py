"""
Fully commented, end-to-end implementation that *projects* a desired (possibly
infeasible) state trajectory onto the dynamics manifold using iLQR.

Key idea (what the code does):
  • You provide a desired/target trajectory x_diff[0:T] (e.g., from diffusion).
  • iLQR finds controls U[0:T-1] such that the rolled-out trajectory X is
    dynamically feasible and stays close to x_diff by minimizing

        J = Σ_t (x_t - x_diff_t)^T Q (x_t - x_diff_t) + u_t^T R u_t
            + (x_T - x_diff_T)^T Qf (x_T - x_diff_T)

  • Dynamics are treated as a *black box* step function (x,u) -> x_next.
    We linearize that black box via finite differences to compute A_t,B_t.

Includes a minimal pendulum example with RK4 stepper.
Swap `blackbox_step` with your own simulator to use on another system.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

Array = np.ndarray

# ======================================================================
# 1) EXAMPLE BLACK-BOX DYNAMICS — PENDULUM (continuous time + RK4 to DT)
# ======================================================================
# State x = [theta, omega]  (angle [rad], angular velocity [rad/s])
# Control u = torque [Nm]
# We expose a *discrete-time* black-box step: x_{t+1} = blackbox_step(x_t,u_t,dt,params)
# For the example we build it by integrating a continuous-time model with RK4.


def pendulum_dynamics(x: Array, u: Array, params: dict) -> Array:
    """Continuous-time ODE: xdot = f(x,u).

    Args:
        x: array([theta, omega])
        u: scalar torque (array-like with shape (1,) or ())
        params: dict with keys g,l,m,b for gravity, length, mass, damping
    Returns:
        xdot = [omega, theta_ddot]
    """
    g = params.get("g", 9.81)
    l = params.get("l", 1.0)
    m = params.get("m", 1.0)
    b = params.get("b", 0.05)  # viscous friction

    theta, omega = x
    torque = float(u)

    # EoM: m l^2 theta_ddot + b omega + m g l sin(theta) = torque
    theta_ddot = (torque - b*omega - m*g*l*np.sin(theta)) / (m*l*l)
    return np.array([omega, theta_ddot], dtype=float)


def rk4_step(f_ct: Callable[[Array, Array, dict], Array],
             x: Array,
             u: Array,
             dt: float,
             params: dict) -> Array:
    """One fixed-step RK4 integration of xdot = f_ct(x,u).

    This turns the continuous-time model into a discrete-time step.
    """
    k1 = f_ct(x, u, params)
    k2 = f_ct(x + 0.5*dt*k1, u, params)
    k3 = f_ct(x + 0.5*dt*k2, u, params)
    k4 = f_ct(x + dt*k3, u, params)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def blackbox_step(x: Array, u: Array, dt: float, params: dict) -> Array:
    """Our example *black-box* discrete update.
    Replace this with your own simulator: x_{t+1} = f(x_t, u_t).
    """
    return rk4_step(pendulum_dynamics, x, u, dt, params)


# ======================================================================
# 2) FINITE-DIFFERENCE LINEARIZATION OF THE BLACK-BOX STEP
# ======================================================================
# Given x,u, approximate A = ∂f/∂x and B = ∂f/∂u using forward differences.


def fd_jacobians(f_step: Callable[[Array, Array, float, dict], Array],
                 x: Array,
                 u: Array,
                 dt: float,
                 params: dict,
                 eps_x: float = 1e-5,
                 eps_u: float = 1e-5) -> Tuple[Array, Array, Array]:
    """Compute discrete-time Jacobians A,B around (x,u) for the map f_step.

    Returns
    -------
    A : (n,n)  Jacobian wrt state
    B : (n,m)  Jacobian wrt control
    fxu : (n,) f_step(x,u,dt,params) (cached forward eval)
    """
    n = x.size
    m = u.size if np.ndim(u) > 0 else 1
    x = x.astype(float)
    u = np.array(u, dtype=float).reshape(m)

    fxu = f_step(x, u, dt, params)
    A = np.zeros((n, n), dtype=float)
    B = np.zeros((n, m), dtype=float)

    # --- ∂f/∂x via coordinate-wise perturbation
    for i in range(n):
        dx = np.zeros_like(x)
        # scale epsilon with magnitude to avoid catastrophic cancellation
        dx[i] = eps_x if abs(x[i]) < 1.0 else eps_x*max(1.0, abs(x[i]))
        f_plus = f_step(x + dx, u, dt, params)
        A[:, i] = (f_plus - fxu) / dx[i]

    # --- ∂f/∂u similarly
    for j in range(m):
        du = np.zeros_like(u)
        du[j] = eps_u if abs(u[j]) < 1.0 else eps_u*max(1.0, abs(u[j]))
        f_plus = f_step(x, u + du, dt, params)
        B[:, j] = (f_plus - fxu) / du[j]

    return A, B, fxu


# ======================================================================
# 3) iLQR CORE — PROJECT TO DYNAMICS MANIFOLD WHILE TRACKING x_diff
# ======================================================================

@dataclass
class ILQRConfig:
    """Configuration for iLQR projection.

    Q, Qf: state tracking weights (positive semidefinite).
    R: control regularization (positive definite recommended).
    reg_*: Levenberg–Marquardt regularization for Quu to guarantee PD.
    """
    Q: Array
    R: Array
    Qf: Array
    max_iters: int = 800
    tol: float = 1e-6
    reg_init: float = 1e-6
    reg_factor: float = 10.0
    umin: Optional[Array] = None  # elementwise bounds (shape (m,))
    umax: Optional[Array] = None
    line_search_alphas: Tuple[float, ...] = (1.0, 0.6, 0.3, 0.1, 0.03, 0.01)


def rollout(f_step, x0: Array, U: Array, dt: float, params: dict) -> Array:
    """Roll out the black-box dynamics to build a *feasible* trajectory X.

    Note: feasibility is ensured because we always use f_step to compute x_{t+1}.
    """
    T = U.shape[0]
    n = x0.size
    X = np.zeros((T+1, n), dtype=float)
    X[0] = x0
    x = x0.copy()
    for t in range(T):
        x = f_step(x, U[t], dt, params)
        X[t+1] = x
    return X


def clamp_u(u: Array, umin: Optional[Array], umax: Optional[Array]) -> Array:
    """Apply simple box constraints to control (optional)."""
    if umin is not None:
        u = np.maximum(u, umin)
    if umax is not None:
        u = np.minimum(u, umax)
    return u


def ilqr_project_to_dynamics(
    x_diff: Array,               # (T+1, n) desired states from diffusion
    x0: Array,                   # initial state (should equal x_diff[0])
    U_init: Array,               # (T, m) initial control sequence (e.g., zeros)
    dt: float,
    params: dict,
    f_step: Callable[[Array, Array, float, dict], Array] = blackbox_step,
    cfg: Optional[ILQRConfig] = None,
) -> Tuple[Array, Array, dict]:
    """Run iLQR to find a feasible trajectory X,U that stays close to x_diff.

    Returns
    -------
    X : (T+1,n) dynamics-consistent trajectory
    U : (T,m)   controls that realize X
    info : diagnostics (final cost, iterations, reg)
    """
    if cfg is None:
        # Safe defaults if the user didn't pass a config
        n = x_diff.shape[1]
        m = U_init.shape[1]
        cfg = ILQRConfig(Q=np.eye(n), R=1e-3*np.eye(m), Qf=10*np.eye(n))

    T, m = U_init.shape
    n = x_diff.shape[1]

    # Initialize with provided controls and a feasible rollout
    U = U_init.copy()
    X = rollout(f_step, x0, U, dt, params)

    def cost(X: Array, U: Array) -> float:
        """Quadratic tracking cost J(X,U)."""
        J = 0.0
        for t in range(T):
            dx = X[t] - x_diff[t]
            du = U[t]
            J += dx @ cfg.Q @ dx + du @ cfg.R @ du
        dxT = X[-1] - x_diff[-1]
        J += dxT @ cfg.Qf @ dxT
        return float(J)

    mu = cfg.reg_init                # LM regularization strength
    J_prev = cost(X, U)              # current cost

    for it in range(cfg.max_iters):
        # --------------------------------------------------------------
        # (a) Linearize dynamics along current (X,U) using finite diff.
        # --------------------------------------------------------------
        A = np.zeros((T, n, n))
        B = np.zeros((T, n, m))
        for t in range(T):
            At, Bt, _ = fd_jacobians(f_step, X[t], U[t], dt, params)
            A[t], B[t] = At, Bt

        # --------------------------------------------------------------
        # (b) Backward pass to compute feedback K_t and feedforward k_t
        # --------------------------------------------------------------
        Vx  = 2.0 * (cfg.Qf @ (X[-1] - x_diff[-1]))
        Vxx = 2.0 * cfg.Qf
        Ks = np.zeros((T, m, n))  # feedback gains
        ks = np.zeros((T, m))     # feedforward terms

        diverged = False
        for t in reversed(range(T)):
            # Quadratic expansion of the Q-function at time t
            Qx  = 2.0*cfg.Q @ (X[t] - x_diff[t]) + A[t].T @ Vx
            Qu  = 2.0*cfg.R @ U[t] + B[t].T @ Vx
            Qxx = 2.0*cfg.Q + A[t].T @ Vxx @ A[t]
            Quu = 2.0*cfg.R + B[t].T @ Vxx @ B[t]
            Qux = B[t].T @ Vxx @ A[t]

            # LM regularization to ensure Quu is positive-definite
            Quu_reg = Quu + mu * np.eye(m)
            try:
                L = np.linalg.cholesky(Quu_reg)
                Linv = np.linalg.inv(L)
                Quu_inv = Linv.T @ Linv
            except np.linalg.LinAlgError:
                # Not PD: increase mu and re-do backward pass
                diverged = True
                break

            # Optimal local policy: u = u_bar + k + K (x - x_bar)
            K = - Quu_inv @ Qux
            k = - Quu_inv @ Qu
            Ks[t], ks[t] = K, k

            # Update value function for next step in the backward sweep
            Vx  = Qx  + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
            Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
            Vxx = 0.5 * (Vxx + Vxx.T)  # numerically symmetrize

        if diverged:
            mu *= cfg.reg_factor
            continue

        # --------------------------------------------------------------
        # (c) Forward pass with line search to update (X,U)
        # --------------------------------------------------------------
        accepted = False
        for alpha in cfg.line_search_alphas:
            X_new = np.zeros_like(X)
            U_new = np.zeros_like(U)
            X_new[0] = x0.copy()

            ok = True
            for t in range(T):
                # Affine policy around current nominal (X,U)
                u = U[t] + alpha*ks[t] + Ks[t] @ (X_new[t] - X[t])
                u = clamp_u(u, cfg.umin, cfg.umax)
                U_new[t] = u

                # Roll the *true* black-box dynamics (feasibility enforced)
                X_new[t+1] = f_step(X_new[t], u, dt, params)
                if not np.all(np.isfinite(X_new[t+1])):
                    ok = False
                    break
            if not ok:
                continue

            J_new = cost(X_new, U_new)
            if J_new < J_prev:  # sufficient decrease found
                X, U, J_prev = X_new, U_new, J_new
                mu = max(cfg.reg_init, mu / cfg.reg_factor)
                accepted = True
                break

        if not accepted:
            # Step rejected: increase regularization and retry
            mu *= cfg.reg_factor

        # Convergence test: small feedforward step
        if accepted and (np.linalg.norm(ks, ord=np.inf) < cfg.tol):
            break

    info = {"final_cost": J_prev, "iters": it+1, "reg": mu}
    return X, U, info


# ======================================================================
# 4) UTILITIES FOR VALIDATION / DEMO
# ======================================================================

def max_dynamics_residual(X: Array, U: Array, dt: float, params: dict,
                          f_step: Callable[[Array, Array, float, dict], Array]) -> float:
    """Compute max || X[t+1] - f_step(X[t],U[t]) || to check feasibility."""
    T = U.shape[0]
    res = 0.0
    for t in range(T):
        pred = f_step(X[t], U[t], dt, params)
        res = max(res, float(np.linalg.norm(X[t+1] - pred)))
    return res


# ======================================================================
# 5) MINIMAL RUNNABLE EXAMPLE (PENDULUM)
# ======================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    # ---- Problem setup
    dt = 0.02
    T = 200
    params = {"g": 9.81, "l": 1.0, "m": 1.0, "b": 0.05}

    # Target (possibly infeasible) trajectory from some generator (e.g., diffusion)
    tgrid = np.arange(T+1) * dt
    theta_ref = np.pi * (1 - np.cos(np.pi * tgrid / (T*dt)))  # smooth sweep 0 -> 2π
    omega_ref = np.gradient(theta_ref, dt)
    x_diff = np.stack([theta_ref, omega_ref], axis=1)  # shape (T+1,2)

    # Initial state (should match x_diff[0] if you want zero initial mismatch)
    x0 = x_diff[0]

    # Initial control guess (zeros are fine)
    U0 = np.zeros((T, 1))

    # Tracking/effort weights (tune these!)
    Q  = np.diag([5.0, 0.1])
    R  = np.diag([1e-3])
    Qf = np.diag([20.0, 2.0])

    cfg = ILQRConfig(
        Q=Q, R=R, Qf=Qf,
        umin=np.array([-5.0]), umax=np.array([5.0]),  # torque limits
        max_iters=150,
    )

    # ---- Run the projection
    X_proj, U_proj, info = ilqr_project_to_dynamics(
        x_diff=x_diff,
        x0=x0,
        U_init=U0,
        dt=dt,
        params=params,
        f_step=blackbox_step,
        cfg=cfg,
    )

    # Diagnostics
    print("iLQR finished:", info)
    print("Final tracking error (Frobenius):", np.linalg.norm(X_proj - x_diff))
    print("Max dynamics residual:", max_dynamics_residual(X_proj, U_proj, dt, params, blackbox_step))
