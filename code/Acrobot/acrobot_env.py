"""
Created on Thu Oct 16 14:20:00 2025

@author: Vivek Sharma
Dynamics follow the standard 2-link manipulator (Acrobot) with optional
underactuation (actuation only at joint 2).
"""

import numpy as np
import torch

from Acrobot.plots import plot_traj, traj_comparison  # keep identical API

class AcrobotEnv():
    """
    This environment describes a planar 2-link acrobot (double pendulum).
    State: [q1, q2, q1_dot, q2_dot] with revolute joints.

    ## Action Space
    If underactuated=True (default): only joint 2 is actuated.
    | Num | Action                  | Min  | Max | Name | Unit |
    | --- | ----------------------- | ---- | --- | ---- | ---- |
    | 0   | Torque at joint 2       | -umax| umax|  u2  |  Nm  |

    If underactuated=False (fully actuated):
    | Num | Action       | Min  | Max | Name | Unit |
    | --- | ------------ | ---- | --- | ---- | ---- |
    | 0   | Torque at j1 | -umax| umax|  u1  |  Nm  |
    | 1   | Torque at j2 | -umax| umax|  u2  |  Nm  |

    ## Observation Space
    | Num | Observation      | Min    | Max    | Name     | Unit |
    | --- | ---------------- | ------ | ------ | -------- | ---- |
    | 0   | Joint 1 angle    | -pi    |  pi    |   q1     |  rad |
    | 1   | Joint 2 angle    | -pi    |  pi    |   q2     |  rad |
    | 2   | Joint 1 vel.     | -wmax  |  wmax  |  q1_dot  | rad/s|
    | 3   | Joint 2 vel.     | -wmax  |  wmax  |  q2_dot  | rad/s|

    ## Starting State
    Near the downward rest with small Gaussian noise of magnitude `reset_noise_scale`.

    ## Episode End
    1. Any state leaves the configured bounds (angles wrap, velocities bounded).
    2. (Optional) If angle error is sufficiently small for a while (not used by default).

    Notes:
    - Angles wrap to (-pi, pi] after each step.
    - Reward is shaped for swing-up by penalizing angle error to upright and velocities,
      plus small control penalty. Adjust as desired.
    """

    def __init__(self,
                 m1: float = 1.0,
                 m2: float = 1.0,
                 l1: float = 1.0,
                 l2: float = 1.0,
                 gravity: float = 9.81,
                 underactuated: bool = True,
                 u_max: float = 5.0,
                 w_max: float = 20.0,
                 reset_noise_scale: float = 1e-2,
                 dt: float = 0.01):
        # Basic info
        self.name = "Acrobot"
        self.state_size = 4
        self.underactuated = underactuated

        self.action_size = 1 if underactuated else 2
        self.action_min = np.array([[-u_max] if underactuated else [-u_max, -u_max]])
        self.action_max = np.array([[ u_max] if underactuated else [ u_max,  u_max]])

        # State indexing groups
        self.position_states = [0, 1]
        self.velocity_states = [2, 3]

        # Physical parameters
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.gravity = gravity

        # Centers of mass and inertias (about joint axes)
        self.lc1 = l1 / 2.0
        self.lc2 = l2 / 2.0
        self.I1 = m1 * (l1 ** 2)   # as given
        self.I2 = m2 * (l2 ** 2)   # as given

        # Time
        self.dt = dt
        self.t = 0.0

        # Bounds
        self.u_max = u_max
        self.w_max = w_max
        # Angles are wrapped, but keep bounds for termination checks (vel limits dominate)
        self.low_bound  = np.array([-np.pi, -np.pi, -w_max, -w_max])
        self.high_bound = np.array([ np.pi,  np.pi,  w_max,  w_max])

        # Reset noise
        self.reset_noise = reset_noise_scale

        # Target (upright): q1 + q2 = pi, e.g., [pi, 0] is straight-up
        self.target_angles = np.array([np.pi, 0.0])

        # Default initial state near downward (0,0,0,0)
        self.x0 = np.array([0.0, 0.0, 0.0, 0.0])

        # Initialize state
        self.state = None
        self.reset()

    # ---------------------------
    # Utilities
    # ---------------------------
    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Wrap angle to (-pi, pi]."""
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _wrap_angles_inplace(self):
        self.state[0] = self._wrap_angle(self.state[0])
        self.state[1] = self._wrap_angle(self.state[1])

    # ---------------------------
    # Manipulator dynamics
    # ---------------------------
    def get_manipulator_matrices(self, x: np.ndarray):
        """
        Calculates the manipulator matrices M, C, tau_g and B.
        Also returns the inverse of M.

        Follows your provided formulas exactly.
        """
        q1, q2, q1_dot, q2_dot = x
        m1, m2 = self.m1, self.m2
        I1, I2 = self.I1, self.I2
        l1, l2 = self.l1, self.l2
        lc1, lc2 = self.lc1, self.lc2
        g = self.gravity

        M = np.array([
            [I1 + I2 + m2 * l1 ** 2 + 2 * m2 * l1 * lc2 * np.cos(q2), I2 + m2 * l1 * lc2 * np.cos(q2)],
            [I2 + m2 * l1 * lc2 * np.cos(q2), I2]
        ])

        # 2x2 inverse
        M_inv = np.array([
            [M[1, 1], -M[0, 1]],
            [-M[1, 0], M[0, 0]]
        ]) / (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])

        C = np.array([
            [-2 * m2 * l1 * lc2 * np.sin(q2) * q2_dot, -m2 * l1 * lc2 * np.sin(q2) * q2_dot],
            [ m2 * l1 * lc2 * np.sin(q2) * q1_dot, 0.0]
        ])

        tau_g = np.array([
            [-m1 * g * lc1 * np.sin(q1) - m2 * g * (l1 * np.sin(q1) + lc2 * np.sin(q1 + q2))],
            [ -m2 * g * lc2 * np.sin(q1 + q2)]
        ])

        if self.underactuated:
            B = np.array([[0.0],
                          [1.0]])
        else:
            B = np.eye(2)

        return M, M_inv, C, tau_g, B

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray):
        """
        Continuous-time dynamics: x_dot = f(x) + g(x) u with manipulator form.
        x = [q1, q2, q1_dot, q2_dot], u shape = (1,) or (2,)
        """
        q_dot = np.array([[x[2]], [x[3]]])  # 2x1

        M, M_inv, C, tau_g, B = self.get_manipulator_matrices(x)

        # Ensure u is column vector with correct size
        u = np.atleast_1d(u)
        if self.underactuated:
            assert u.shape == (1,), f"Expected u shape (1,) for underactuated, got {u.shape}"
            u_vec = u.reshape(1, 1)  # (1,1)
        else:
            assert u.shape == (2,), f"Expected u shape (2,) for fully actuated, got {u.shape}"
            u_vec = u.reshape(2, 1)  # (2,1)

        qdd = M_inv.dot(tau_g + B.dot(u_vec) - C.dot(q_dot))  # 2x1
        x1_dot = x[2]
        x2_dot = x[3]
        x3_dot = qdd[0, 0]
        x4_dot = qdd[1, 0]
        return np.array([x1_dot, x2_dot, x3_dot, x4_dot])

    # ---------------------------
    # Gym-like API
    # ---------------------------
    def reset(self):
        """Reset near the downward configuration with small noise."""
        self.state = self.x0.copy() + self.reset_noise * np.random.randn(self.state_size)
        self._wrap_angles_inplace()
        self.t = 0.0
        return self.state.copy()

    def reset_to(self, state: np.ndarray):
        """Set state explicitly (angles will be wrapped to (-pi, pi])."""
        assert state.shape == (4,)
        self.state = state.copy()
        self._wrap_angles_inplace()
        return self.state.copy()

    def step(self, u: np.ndarray):
        """
        One Euler step forward with torque saturation and angle wrapping.
        Returns: obs, reward, terminated, truncated, info
        """
        # Saturate action
        u = np.clip(u, self.action_min, self.action_max).ravel()

        # Integrate
        xd = self.dynamics(self.t, self.state, u)
        self.state = self.state + self.dt * xd
        self.t += self.dt

        # Wrap angles and clip velocities
        self._wrap_angles_inplace()
        self.state[2:] = np.clip(self.state[2:], -self.w_max, self.w_max)

        # Termination if velocities hit limits (you can relax this)
        out_of_bound = np.any(self.state < self.low_bound) or np.any(self.state > self.high_bound)

        # Reward shaping for swing-up (angle error to upright + velocity & control penalties)
        # Angle distance considering wrapping
        def ang_err(a, a_ref):
            return np.arctan2(np.sin(a - a_ref), np.cos(a - a_ref))
        e1 = ang_err(self.state[0], self.target_angles[0])
        e2 = ang_err(self.state[1], self.target_angles[1])
        angle_cost = e1**2 + e2**2
        vel_cost = 0.01 * (self.state[2]**2 + self.state[3]**2)
        ctrl_cost = 0.001 * float(u @ u)
        reward = 1.0 - angle_cost - vel_cost - ctrl_cost - 1.0 * out_of_bound

        terminated = bool(out_of_bound)
        truncated = False
        info = None
        return self.state.copy(), reward, terminated, truncated, info

    # ---------------------------
    # Projector-compatible helper
    # ---------------------------
    def pos_from_vel(self, S_t: torch.Tensor, vel_t_dt: torch.Tensor):
        """
        Calculates next state's *position* using explicit Euler on angles only.
        This mirrors the QuadcopterEnv.pos_from_vel signature/behavior:
          - S_t: current state (torch.tensor, shape (4,))
          - vel_t_dt: (unused) next state's velocity tensor
        Returns:
          - x_t_dt: next state's position tensor (q1, q2) (torch.tensor, shape (2,))
        """
        q1 = S_t[0]
        q2 = S_t[1]
        q1_dot = S_t[2]
        q2_dot = S_t[3]

        q_next = torch.stack([q1 + self.dt * q1_dot,
                              q2 + self.dt * q2_dot])

        # Wrap to (-pi, pi] in PyTorch
        pi = torch.tensor(np.pi, dtype=q_next.dtype, device=q_next.device)
        q_next = (q_next + pi) % (2 * pi) - pi
        return q_next

    # ---------------------------
    # Plotting functions (same API)
    # ---------------------------
    def plot_traj(self, Traj: np.ndarray, title: str = ""):
        """Plot trajectory (e.g., joint angles or end-effector path) via external helper."""
        plot_traj(self, Traj, title)

    def traj_comparison(self, traj_1, label_1, traj_2, label_2, title: str = "",
                        traj_3=None, label_3=None, traj_4=None, label_4=None,
                        legend_loc='best'):
        """
        Compare up to 4 trajectories (H, 4) using the same helper signature as QuadcopterEnv.
        """
        traj_comparison(self, traj_1, label_1, traj_2, label_2, title,
                        traj_3, label_3, traj_4, label_4, legend_loc)



