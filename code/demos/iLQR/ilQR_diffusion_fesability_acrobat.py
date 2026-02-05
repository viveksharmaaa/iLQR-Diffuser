"""
Project a desired (possibly infeasible) *state* trajectory onto the dynamics
manifold using iLQR for the **Acrobot** (two-link underactuated pendulum).

- State x = [th1, th2, dth1, dth2]. Angles in radians.
- Control u = [tau2] (actuation at the elbow only; shoulder is passive).
- Dynamics are treated as a *black-box* single-step function. We build it by
  integrating a continuous-time model with RK4.
- Linearization for iLQR uses finite differences on the black-box step.

The core iLQR implementation is generic and can be reused for other systems.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

Array = np.ndarray

# ======================================================================
# 1) ACROBOT CONTINUOUS-TIME DYNAMICS + RK4 → DISCRETE BLACK-BOX STEP
# ======================================================================
# Model adapted from classic Acrobot equations (Spong), using standard
# parameters; configurable via `params` dict.
# x = [th1, th2, dth1, dth2], u = [tau2]


def wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi] to prevent drift; optional in stepper."""
    return (a + np.pi) % (2*np.pi) - np.pi


def acrobot_ct_dynamics(x: Array, u: Array, params: dict) -> Array:
    """Continuous-time Acrobot ODE: xdot = f(x,u).

    Parameters (with typical defaults if absent):
      m1,m2  : link masses
      l1,l2  : link lengths
      lc1,lc2: COM lengths (from proximal joint)
      I1,I2  : link inertias about COM
      b1,b2  : viscous damping
      g      : gravity
    """
    m1 = params.get("m1", 1.0)
    m2 = params.get("m2", 1.0)
    l1 = params.get("l1", 1.0)
    l2 = params.get("l2", 1.0)
    lc1 = params.get("lc1", 0.5*l1)
    lc2 = params.get("lc2", 0.5*l2)
    I1 = params.get("I1", 1.0/12.0)  # rod about COM
    I2 = params.get("I2", 1.0/12.0)
    b1 = params.get("b1", 0.01)
    b2 = params.get("b2", 0.01)
    g  = params.get("g", 9.81)

    th1, th2, dth1, dth2 = x
    tau2 = float(u)  # single input at elbow

    c2 = np.cos(th2)
    s2 = np.sin(th2)

    # Inertia matrix terms (Spong)
    d11 = I1 + I2 + m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*c2)
    d12 = I2 + m2*(lc2**2 + l1*lc2*c2)
    d21 = d12
    d22 = I2 + m2*lc2**2

    # Coriolis/Centrifugal
    h = -m2*l1*lc2*s2
    c1 = h * (2*dth1*dth2 + dth2**2)
    c2_term = h * dth1**2

    # Gravity
    phi1 = (m1*lc1 + m2*l1)*g*np.cos(th1) + m2*lc2*g*np.cos(th1+th2)
    phi2 = m2*lc2*g*np.cos(th1+th2)

    # Damping and input (u acts on joint 2)
    # Equations: D(q) * ddq + C(q,qd) + B*qd + G(q) = [0; tau2]
    D = np.array([[d11, d12], [d21, d22]])
    C = np.array([c1 + b1*dth1, c2_term + b2*dth2])
    G = np.array([phi1, phi2])
    tau = np.array([0.0, tau2])

    # Solve for accelerations
    dd = np.linalg.solve(D, tau - C - G)
    ddth1, ddth2 = dd

    return np.array([dth1, dth2, ddth1, ddth2], dtype=float)


def rk4_step(f_ct: Callable[[Array, Array, dict], Array],
             x: Array,
             u: Array,
             dt: float,
             params: dict) -> Array:
    """One-step RK4 integrator for xdot = f_ct(x,u)."""
    k1 = f_ct(x, u, params)
    k2 = f_ct(x + 0.5*dt*k1, u, params)
    k3 = f_ct(x + 0.5*dt*k2, u, params)
    k4 = f_ct(x + dt*k3, u, params)
    xn = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    # optional angle wrapping for numerical hygiene
    xn[0] = wrap_angle(xn[0])
    xn[1] = wrap_angle(xn[1])
    return xn


def blackbox_step_acrobot(x: Array, u: Array, dt: float, params: dict) -> Array:
    return rk4_step(acrobot_ct_dynamics, x, u, dt, params)


# ======================================================================
# 2) GENERIC FD LINEARIZATION AND I-LQR (REUSED FROM PENDULUM VERSION)
# ======================================================================

def fd_jacobians(f_step: Callable[[Array, Array, float, dict], Array],
                 x: Array,
                 u: Array,
                 dt: float,
                 params: dict,
                 eps_x: float = 1e-5,
                 eps_u: float = 1e-5):
    n = x.size
    m = u.size if np.ndim(u) > 0 else 1
    x = x.astype(float)
    u = np.array(u, dtype=float).reshape(m)

    fxu = f_step(x, u, dt, params)
    A = np.zeros((n, n), dtype=float)
    B = np.zeros((n, m), dtype=float)

    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = eps_x if abs(x[i]) < 1.0 else eps_x*max(1.0, abs(x[i]))
        f_plus = f_step(x + dx, u, dt, params)
        A[:, i] = (f_plus - fxu) / dx[i]

    for j in range(m):
        du = np.zeros_like(u)
        du[j] = eps_u if abs(u[j]) < 1.0 else eps_u*max(1.0, abs(u[j]))
        f_plus = f_step(x, u + du, dt, params)
        B[:, j] = (f_plus - fxu) / du[j]

    return A, B, fxu


@dataclass
class ILQRConfig:
    Q: Array
    R: Array
    Qf: Array
    max_iters: int = 200
    tol: float = 1e-6
    reg_init: float = 1e-6
    reg_factor: float = 10.0
    umin: Optional[Array] = None
    umax: Optional[Array] = None
    line_search_alphas: tuple = (1.0, 0.6, 0.3, 0.1, 0.03, 0.01)


def rollout(f_step, x0: Array, U: Array, dt: float, params: dict) -> Array:
    T = U.shape[0]
    n = x0.size
    X = np.zeros((T+1, n), dtype=float)
    X[0] = x0
    for t in range(T):
        X[t+1] = f_step(X[t], U[t], dt, params)
    return X


def clamp_u(u: Array, umin: Optional[Array], umax: Optional[Array]) -> Array:
    if umin is not None:
        u = np.maximum(u, umin)
    if umax is not None:
        u = np.minimum(u, umax)
    return u


def ilqr_project_to_dynamics(
    x_diff: Array,
    x0: Array,
    U_init: Array,
    dt: float,
    params: dict,
    f_step: Callable[[Array, Array, float, dict], Array] = blackbox_step_acrobot,
    cfg: Optional[ILQRConfig] = None,
):
    if cfg is None:
        n = x_diff.shape[1]
        m = U_init.shape[1]
        cfg = ILQRConfig(Q=np.eye(n), R=1e-3*np.eye(m), Qf=10*np.eye(n))

    T, m = U_init.shape
    n = x_diff.shape[1]

    U = U_init.copy()
    X = rollout(f_step, x0, U, dt, params)

    def cost(X: Array, U: Array) -> float:
        J = 0.0
        for t in range(T):
            dx = X[t] - x_diff[t]
            du = U[t]
            J += dx @ cfg.Q @ dx + du @ cfg.R @ du
        dxT = X[-1] - x_diff[-1]
        J += dxT @ cfg.Qf @ dxT
        return float(J)

    mu = cfg.reg_init
    J_prev = cost(X, U)

    for it in range(cfg.max_iters):
        # Linearize along current trajectory
        A = np.zeros((T, n, n))
        B = np.zeros((T, n, m))
        for t in range(T):
            At, Bt, _ = fd_jacobians(f_step, X[t], U[t], dt, params)
            A[t], B[t] = At, Bt

        # Backward pass
        Vx  = 2.0 * (cfg.Qf @ (X[-1] - x_diff[-1]))
        Vxx = 2.0 * cfg.Qf
        Ks = np.zeros((T, m, n))
        ks = np.zeros((T, m))

        diverged = False
        for t in reversed(range(T)):
            Qx  = 2.0*cfg.Q @ (X[t] - x_diff[t]) + A[t].T @ Vx
            Qu  = 2.0*cfg.R @ U[t] + B[t].T @ Vx
            Qxx = 2.0*cfg.Q + A[t].T @ Vxx @ A[t]
            Quu = 2.0*cfg.R + B[t].T @ Vxx @ B[t]
            Qux = B[t].T @ Vxx @ A[t]

            Quu_reg = Quu + mu*np.eye(m)
            try:
                L = np.linalg.cholesky(Quu_reg)
                Linv = np.linalg.inv(L)
                Quu_inv = Linv.T @ Linv
            except np.linalg.LinAlgError:
                diverged = True
                break

            K = - Quu_inv @ Qux
            k = - Quu_inv @ Qu
            Ks[t], ks[t] = K, k

            Vx  = Qx  + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
            Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
            Vxx = 0.5*(Vxx + Vxx.T)

        if diverged:
            mu *= cfg.reg_factor
            continue

        # Forward pass + line search
        accepted = False
        for alpha in cfg.line_search_alphas:
            Xn = np.zeros_like(X)
            Un = np.zeros_like(U)
            Xn[0] = x0.copy()
            ok = True
            for t in range(T):
                u = U[t] + alpha*ks[t] + Ks[t] @ (Xn[t] - X[t])
                u = clamp_u(u, cfg.umin, cfg.umax)
                Un[t] = u
                Xn[t+1] = f_step(Xn[t], u, dt, params)
                if not np.all(np.isfinite(Xn[t+1])):
                    ok = False
                    break
            if not ok:
                continue
            Jn = cost(Xn, Un)
            if Jn < J_prev:
                X, U, J_prev = Xn, Un, Jn
                mu = max(cfg.reg_init, mu/cfg.reg_factor)
                accepted = True
                break
        if not accepted:
            mu *= cfg.reg_factor
        if accepted and (np.linalg.norm(ks, ord=np.inf) < cfg.tol):
            break

    info = {"final_cost": J_prev, "iters": it+1, "reg": mu}
    return X, U, info


# ======================================================================
# 3) SMALL DEMO: create a target x_diff and project to feasible (swing-up)
# ======================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    # Horizon and time-step
    dt = 0.02
    T = 400

    # Default Acrobot parameters
    params = {
        "m1": 1.0, "m2": 1.0,
        "l1": 1.0, "l2": 1.0,
        "lc1": 0.5, "lc2": 0.5,
        "I1": 1.0/12.0, "I2": 1.0/12.0,
        "b1": 0.01, "b2": 0.01,
        "g": 9.81,
    }

    # Fabricate a smooth target trajectory that goes near the upright (pi, 0)
    t = np.arange(T+1)*dt
    th1_ref = np.pi * (1 - np.cos(np.pi * t / (T*dt)))  # 0 -> 2pi smooth
    th2_ref = 0.5*np.sin(2*np.pi * t / (T*dt))          # small elbow swing
    dth1_ref = np.gradient(th1_ref, dt)
    dth2_ref = np.gradient(th2_ref, dt)
    x_diff = np.stack([th1_ref, th2_ref, dth1_ref, dth2_ref], axis=1)

    # Initial state = start of target (usually hanging down)
    x0 = x_diff[0]

    # Initial controls (zeros)
    U0 = np.zeros((T, 1))

    # Weights: track angles more than velocities; small control penalty
    Q  = np.diag([10.0, 5.0, 0.1, 0.1])
    R  = np.diag([1e-3])
    Qf = np.diag([50.0, 25.0, 1.0, 1.0])

    cfg = ILQRConfig(Q=Q, R=R, Qf=Qf,
                     umin=np.array([-6.0]), umax=np.array([6.0]),
                     max_iters=250)

    X_proj, U_proj, info = ilqr_project_to_dynamics(
        x_diff=x_diff,
        x0=x0,
        U_init=U0,
        dt=dt,
        params=params,
        f_step=blackbox_step_acrobot,
        cfg=cfg,
    )

    # Report
    def max_dyn_res(X, U):
        res = 0.0
        for k in range(U.shape[0]):
            pred = blackbox_step_acrobot(X[k], U[k], dt, params)
            res = max(res, float(np.linalg.norm(X[k+1] - pred)))
        return res

    print("iLQR finished:", info)
    print("Final tracking error ||X - x_diff||_F:", np.linalg.norm(X_proj - x_diff))
    print("Max dynamics residual:", max_dyn_res(X_proj, U_proj))
