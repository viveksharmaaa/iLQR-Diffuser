"""
Generic iLQR projection for **Gymnasium MuJoCo** environments.

Goal: Given a *desired* (possibly infeasible) state trajectory x_diff[0:T]
(from diffusion, planner, etc.), compute controls U[0:T-1] such that the
rollout under the *true* MuJoCo dynamics is feasible AND stays close to x_diff
in the Q-weighted sense.

Key features
------------
- Treats the simulator as a **black box** using the Gymnasium MuJoCo env
  (e.g., "Ant-v4", "HalfCheetah-v4", "Hopper-v4", custom MujocoEnv).
- Computes discrete-time Jacobians (A_t, B_t) via **finite differences** by
  cloning/restoring MuJoCo state so that each derivative call is deterministic.
- Levenberg–Marquardt regularization and backtracking line search for robust iLQR.
- Simple control box constraints (optional).

Assumptions
-----------
- The env follows Gymnasium API and wraps a MuJoCo sim such that you can access
  qpos and qvel, and set them deterministically. This is true for standard
  gymnasium.envs.mujoco.* classes that inherit from MujocoEnv.
- Action space is a Box; observation encodes (at least) qpos/qvel.
- We build a compact *state vector* x := [qpos; qvel]. The helper below maps
  between env <-> x.

Usage (minimal)
---------------
    import gymnasium as gym

    env = gym.make("Hopper-v4", render_mode=None)
    bb = MujocoBlackBox(env)

    T = 200
    dt = bb.dt

    # Build some target trajectory x_diff (T+1, n) — here we just record a
    # rollout under zero controls for illustration. Replace with your diffusion.
    x0 = bb.get_state_vector()
    U0 = np.zeros((T, env.action_space.shape[0]))
    X_target = bb.rollout_from(x0, U0)  # may be feasible; your diffusion need not be

    # iLQR projection
    cfg = ILQRConfig(Q=np.eye(bb.n), R=1e-3*np.eye(bb.m), Qf=10*np.eye(bb.n),
                     umin=env.action_space.low, umax=env.action_space.high)

    X_proj, U_proj, info = ilqr_project_to_dynamics(
        x_diff=X_target, x0=x0, U_init=U0, dt=dt, f_step=bb.f_step, cfg=cfg
    )

    print(info)

Note: To project an *infeasible* diffusion trajectory, set `x_diff = X_from_diffusion`.
Feasibility is enforced by rolling the true env inside iLQR.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import gymnasium as gym
import numpy as np

Array = np.ndarray

# =============================================================
# 1) MuJoCo black-box wrapper: state get/set & single-step map
# =============================================================

class MujocoBlackBox:
    """Black-box interface to a Gymnasium MuJoCo environment.

    State vector x := [qpos; qvel]. We assume deterministic stepping (no noise),
    and provide utilities to clone/restore simulator state for finite differences.
    """
    def __init__(self, env: gym.Env):
        assert hasattr(env, "unwrapped"), "Pass a Gymnasium MuJoCo environment"
        self.env = env
        self.env.reset()
        self.model = env.unwrapped.model
        self.data = env.unwrapped.data
        # Effective dt (MuJoCo base timestep * frame_skip if defined)
        base = float(getattr(self.model, "opt").timestep)
        frame_skip = int(getattr(env.unwrapped, "frame_skip", 1))
        self.dt = base * frame_skip

        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        self.n = self.nq + self.nv
        self.m = int(np.prod(env.action_space.shape))

    # ---------- state helpers ----------
    def get_state_vector(self) -> Array:
        qpos = np.array(self.data.qpos, copy=True).reshape(-1)
        qvel = np.array(self.data.qvel, copy=True).reshape(-1)
        return np.concatenate([qpos, qvel])

    def set_state_vector(self, x: Array) -> None:
        qpos = x[: self.nq]
        qvel = x[self.nq : self.nq + self.nv]
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        # Forward the MuJoCo state for consistency
        import mujoco

        mujoco.mj_forward(self.model, self.data)

    def clone_sim_state(self):
        """Deep-copy MuJoCo core state: qpos, qvel, act, and rng where possible."""
        # MuJoCo core state
        s = {
            "qpos": np.array(self.data.qpos, copy=True),
            "qvel": np.array(self.data.qvel, copy=True),
        }
        # Some envs also have actuator states; copy if present
        if hasattr(self.data, "act"):
            s["act"] = np.array(self.data.act, copy=True)
        # Gymnasium RNG state
        if hasattr(self.env.unwrapped, "np_random"):
            s["rng"] = copy.deepcopy(self.env.unwrapped.np_random.bit_generator.state)
        return s

    def restore_sim_state(self, s):
        self.data.qpos[:] = s["qpos"]
        self.data.qvel[:] = s["qvel"]
        if "act" in s and hasattr(self.data, "act"):
            self.data.act[:] = s["act"]
        if "rng" in s and hasattr(self.env.unwrapped, "np_random"):
            self.env.unwrapped.np_random.bit_generator.state = s["rng"]
        # Forward to refresh derived quantities
        import mujoco

        mujoco.mj_forward(self.model, self.data)

    # ---------- black-box step ----------
    def f_step(self, x: Array, u: Array, dt: float, params: dict | None = None) -> Array:
        """One discrete step of the *true* env from state x and action u.

        We clone&restore state so this call is pure (doesn't affect the live env).
        """
        assert abs(dt - self.dt) < 1e-12, "Use dt=env_dt for consistency"
        snap = self.clone_sim_state()
        try:
            self.set_state_vector(x)
            # Step once using Gymnasium API
            obs, rew, terminated, truncated, info = self.env.step(u.astype(self.env.action_space.dtype))
            x_next = self.get_state_vector()
        finally:
            # Restore original sim state so external code isn't affected
            self.restore_sim_state(snap)
        return x_next

    # ---------- rollout utility ----------
    def rollout_from(self, x0: Array, U: Array) -> Array:
        """Roll the env forward from x0 under controls U, returning X (T+1,n)."""
        snap = self.clone_sim_state()
        try:
            self.set_state_vector(x0)
            T = U.shape[0]
            X = np.zeros((T + 1, self.n))
            X[0] = self.get_state_vector()
            for t in range(T):
                obs, rew, terminated, truncated, info = self.env.step(U[t].astype(self.env.action_space.dtype))
                X[t + 1] = self.get_state_vector()
            return X
        finally:
            self.restore_sim_state(snap)


# =============================================================
# 2) Finite-difference Jacobians for the black-box map
# =============================================================

def fd_jacobians_bb(
    f_step: Callable[[Array, Array, float, Optional[dict]], Array],
    x: Array,
    u: Array,
    dt: float,
    params: Optional[dict] = None,
    eps_x: float = 1e-5,
    eps_u: float = 1e-5,
) -> Tuple[Array, Array, Array]:
    """Finite-difference Jacobians of the discrete map x' = f_step(x,u,dt).

    Returns A (n,n), B (n,m), and f(x,u).
    """
    n = x.size
    m = u.size if np.ndim(u) > 0 else 1
    x = x.astype(float)
    u = np.array(u, dtype=float).reshape(m)

    fxu = f_step(x, u, dt, params)
    A = np.zeros((n, n))
    B = np.zeros((n, m))

    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = eps_x if abs(x[i]) < 1.0 else eps_x * max(1.0, abs(x[i]))
        f_plus = f_step(x + dx, u, dt, params)
        A[:, i] = (f_plus - fxu) / dx[i]

    for j in range(m):
        du = np.zeros_like(u)
        du[j] = eps_u if abs(u[j]) < 1.0 else eps_u * max(1.0, abs(u[j]))
        f_plus = f_step(x, u + du, dt, params)
        B[:, j] = (f_plus - fxu) / du[j]

    return A, B, fxu


# =============================================================
# 3) iLQR core (same structure as before, env-agnostic)
# =============================================================

@dataclass
class ILQRConfig:
    Q: Array
    R: Array
    Qf: Array
    max_iters: int = 150
    tol: float = 1e-6
    reg_init: float = 1e-6
    reg_factor: float = 10.0
    umin: Optional[Array] = None
    umax: Optional[Array] = None
    line_search_alphas: Tuple[float, ...] = (1.0, 0.6, 0.3, 0.1, 0.03, 0.01)


def rollout(f_step, x0: Array, U: Array, dt: float, params: Optional[dict]) -> Array:
    T = U.shape[0]
    n = x0.size
    X = np.zeros((T + 1, n))
    X[0] = x0
    x = x0.copy()
    for t in range(T):
        x = f_step(x, U[t], dt, params)
        X[t + 1] = x
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
    f_step: Callable[[Array, Array, float, Optional[dict]], Array],
    cfg: ILQRConfig,
    params: Optional[dict] = None,
) -> Tuple[Array, Array, dict]:
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
        # Linearize along (X,U)
        A = np.zeros((T, n, n))
        B = np.zeros((T, n, m))
        for t in range(T):
            At, Bt, _ = fd_jacobians_bb(f_step, X[t], U[t], dt, params)
            A[t], B[t] = At, Bt

        # Backward pass
        Vx = 2.0 * (cfg.Qf @ (X[-1] - x_diff[-1]))
        Vxx = 2.0 * cfg.Qf
        Ks = np.zeros((T, m, n))
        ks = np.zeros((T, m))
        diverged = False
        for t in reversed(range(T)):
            Qx = 2.0 * cfg.Q @ (X[t] - x_diff[t]) + A[t].T @ Vx
            Qu = 2.0 * cfg.R @ U[t] + B[t].T @ Vx
            Qxx = 2.0 * cfg.Q + A[t].T @ Vxx @ A[t]
            Quu = 2.0 * cfg.R + B[t].T @ Vxx @ B[t]
            Qux = B[t].T @ Vxx @ A[t]

            Quu_reg = Quu + mu * np.eye(m)
            try:
                L = np.linalg.cholesky(Quu_reg)
                Linv = np.linalg.inv(L)
                Quu_inv = Linv.T @ Linv
            except np.linalg.LinAlgError:
                diverged = True
                break

            K = -Quu_inv @ Qux
            k = -Quu_inv @ Qu
            Ks[t], ks[t] = K, k

            Vx = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
            Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
            Vxx = 0.5 * (Vxx + Vxx.T)

        if diverged:
            mu *= cfg.reg_factor
            continue

        # Forward pass with line search
        accepted = False
        for alpha in cfg.line_search_alphas:
            X_new = np.zeros_like(X)
            U_new = np.zeros_like(U)
            X_new[0] = x0.copy()
            ok = True
            for t in range(T):
                u = U[t] + alpha * ks[t] + Ks[t] @ (X_new[t] - X[t])
                u = clamp_u(u, cfg.umin, cfg.umax)
                U_new[t] = u
                X_new[t + 1] = f_step(X_new[t], u, dt, params)
                if not np.all(np.isfinite(X_new[t + 1])):
                    ok = False
                    break
            if not ok:
                continue
            J_new = cost(X_new, U_new)
            if J_new < J_prev:
                X, U, J_prev = X_new, U_new, J_new
                mu = max(cfg.reg_init, mu / cfg.reg_factor)
                accepted = True
                break
        if not accepted:
            mu *= cfg.reg_factor

        if accepted and (np.linalg.norm(ks, ord=np.inf) < cfg.tol):
            break

    info = {"final_cost": J_prev, "iters": it + 1, "reg": mu}
    return X, U, info


# =============================================================
# 4) Convenience: check dynamics residual using the same black box
# =============================================================

def max_dynamics_residual(
    X: Array, U: Array, dt: float, f_step: Callable[[Array, Array, float, Optional[dict]], Array]
) -> float:
    T = U.shape[0]
    res = 0.0
    for t in range(T):
        pred = f_step(X[t], U[t], dt, None)
        res = max(res, float(np.linalg.norm(X[t + 1] - pred)))
    return res


# =============================================================
# 5) Example main (commented out to avoid running on import)
# =============================================================
if __name__ == "__main__":
    # Example with Hopper-v4 (requires `pip install gymnasium[mujoco] mujoco`)
    env = gym.make("InvertedPendulum-v5")
    bb = MujocoBlackBox(env)

    T = 200
    dt = bb.dt
    x0 = bb.get_state_vector()  # current sim state after reset

    # Build a simple target by rolling zero actions (replace with diffusion output)
    U_zero = np.zeros((T, bb.m))
    X_target = 0.001* bb.rollout_from(x0, U_zero)

    # # Target (possibly infeasible) trajectory from some generator (e.g., diffusion)
    tgrid = np.arange(T+1) * dt
    theta_ref = np.pi * (1 - np.cos(np.pi * tgrid / (T*dt)))  # smooth sweep 0 -> 2π
    omega_ref = np.gradient(theta_ref, dt)
    X_target= np.stack([theta_ref, omega_ref], axis=1)  # shape (T+1,2)


    cfg = ILQRConfig(
        Q=np.eye(bb.n),
        R=1e-3 * np.eye(bb.m),
        Qf=10 * np.eye(bb.n),
        umin=env.action_space.low,
        umax=env.action_space.high,
    )

    X_proj, U_proj, info = ilqr_project_to_dynamics(
        x_diff=X_target,
        x0=x0,
        U_init=U_zero,
        dt=dt,
        f_step=bb.f_step,
        cfg=cfg,
        params=None,
    )

    print("iLQR finished:", info)
    print("Final tracking error (Frob):", np.linalg.norm(X_proj - X_target))
    print("Max dyn residual:", max_dynamics_residual(X_proj, U_proj, dt, bb.f_step))
