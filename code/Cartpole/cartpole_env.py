# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:09:01 2024

@author: Jean-Baptiste Bouvier (style matched)

Cartpole environment written to mirror the QuadcopterEnv API.
"""

import torch
import numpy as np

from Cartpole.plots import plot_traj, traj_comparison


class CartpoleEnv():
    """
    This environment describes a standard planar cart-pole (inverted pendulum on a cart).

    ## Action Space
    | Num | Action                       | Min | Max | Name | Unit |
    | --- | ---------------------------- | --- | --- | ---- | ---- |
    | 0   | Horizontal force on the cart | -20 |  20 |  u   |  N   |

    ## Observation Space
    | Num | Observation                          | Min  | Max | Name   | Unit  |
    | --- | ------------------------------------ | ---- | --- | ------ | ----- |
    | 0   | x-position of the cart               | -Inf | Inf |  x     |  m    |
    | 1   | pole angle (0 = upright, +CCW)       | -Inf | Inf |  theta |  rad  |
    | 2   | x-velocity of the cart               | -Inf | Inf |  x_dot |  m/s  |
    | 3   | pole angular velocity                | -Inf | Inf |  thdot | rad/s |

    ## Starting State
    By default, starts near the upright equilibrium with small Gaussian noise
    of magnitude `reset_noise_scale`.

    ## Episode End
    1. Any state goes out of bounds (x, theta, velocities).
    2. (Optional) The cart leaves a specified track (controlled via bounds).

    NOTES:
    Dynamics follow the standard cart-pole equations used in classic control:
        x_ddot and theta_ddot derived from a simple cart-pendulum with torque-free pivot,
        driven by force u on the cart.
    """

    def __init__(self,
                 reset_noise_scale: float = 1e-2,
                 dt: float = 0.02,
                 x_target: float = 0.0,
                 angle_upright: float = 0.0):
        self.name = "Cartpole"
        self.state_size = 4
        self.action_size = 1

        # Action limits (force on cart)
        self.action_min = np.array([[-20.0]])
        self.action_max = np.array([[ 20.0]])

        # Indices for convenience
        self.position_states = [0, 1]        # x, theta
        self.velocity_states = [2, 3]        # x_dot, theta_dot

        # Target (track center, upright pole)
        self.x_target = float(x_target)
        self.theta_target = float(angle_upright)  # upright

        # Physical parameters (typical values)
        self.g = 9.81        # gravity [m/s^2]
        self.m_c = 1.0       # cart mass [kg]
        self.m_p = 0.1       # pole mass [kg]
        self.l = 0.5         # pole length to COM [m] (i.e., half-length if pole length=1.0)
        self.total_m = self.m_c + self.m_p

        # Bounds (track length, angle/window, velocities)
        self.low_bound  = np.array([-3.0, -np.deg2rad(45.0), -10.0, -10.0])  # [x, theta, xdot, thdot]
        self.high_bound = np.array([ 3.0,  np.deg2rad(45.0),  10.0,  10.0])

        # Integration step
        self.dt = dt
        self.reset_noise = reset_noise_scale

        # Hover/upright-like state near the desired equilibrium (x=target, theta=upright)
        self.upright_state = np.array([self.x_target, self.theta_target, 0.0, 0.0], dtype=float)

        # Internal state buffer
        self.state = self.upright_state.copy()

    # -----------------------
    # Standard environment API
    # -----------------------
    def reset(self):
        noise = self.reset_noise * np.random.randn(self.state_size)
        self.state = (self.upright_state + noise).astype(float)
        # Optionally wrap angle into [-pi, pi] to keep it well-behaved
        self.state[1] = ((self.state[1] + np.pi) % (2 * np.pi)) - np.pi
        return self.state.copy()

    def reset_to(self, state):
        state = np.asarray(state, dtype=float).copy()
        self.state = state
        return self.state.copy()

    def step(self, u):
        """
        One-step dynamics integration with explicit Euler, using standard cart-pole EoMs.

        Args:
            u: array-like with shape (1,) or scalar (force in Newtons)

        Returns:
            next_state (np.ndarray, shape=(4,)),
            reward (float),
            terminated (bool),
            truncated (bool, always False here),
            info (None)
        """
        # Ensure action in array form and clip
        if np.isscalar(u):
            u = np.array([u], dtype=float)
        u = np.asarray(u, dtype=float).reshape(-1)
        u = np.clip(u, self.action_min.ravel(), self.action_max.ravel())
        force = float(u[0])

        x, th, x_dot, th_dot = self.state

        # For readability
        m_c = self.m_c
        m_p = self.m_p
        m_tot = self.total_m
        l = self.l
        g = self.g

        # Standard cart-pole dynamics (same as classic OpenAI Gym formulation)
        # temp = (u + m_p*l*th_dot^2*sin(th)) / (m_c + m_p)
        temp = (force + m_p * l * th_dot * th_dot * np.sin(th)) / m_tot

        # thetaacc = (g*sin(th) - cos(th)*temp) / (l*(4/3 - m_p*cos^2(th)/(m_c+m_p)))
        denom = l * (4.0 / 3.0 - (m_p * np.cos(th) * np.cos(th)) / m_tot)
        th_ddot = (g * np.sin(th) - np.cos(th) * temp) / denom

        # xacc = temp - m_p*l*thetaacc*cos(th)/(m_c+m_p)
        x_ddot = temp - (m_p * l * th_ddot * np.cos(th)) / m_tot

        # Integrate forward (explicit Euler)
        x      = x + self.dt * x_dot
        th     = th + self.dt * th_dot
        x_dot  = x_dot + self.dt * x_ddot
        th_dot = th_dot + self.dt * th_ddot

        # Optionally wrap angle (keeps equivalent states bounded)
        th = ((th + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array([x, th, x_dot, th_dot], dtype=float)

        # Check termination
        out_of_bound = np.any(self.state < self.low_bound) or np.any(self.state > self.high_bound)
        terminated = bool(out_of_bound)  # no obstacles, just bounds
        collided = False  # kept for parity with QuadcopterEnv reward structure

        # Distance to target: cart to x_target and angle to upright (theta_target=0 by default)
        distance_to_target = np.linalg.norm([x - self.x_target, th - self.theta_target])

        # Reward mirrors quadcopter style: encourage staying in-bounds & near target
        reward = 1.0 - float(out_of_bound) - float(collided) - distance_to_target

        return self.state.copy(), reward, terminated, False, None

    # -----------------------
    # Helper used by projectors
    # -----------------------
    def pos_from_vel(self, S_t, vel_t_dt):
        """
        Calculates the next state's *position components* using explicit Euler.
        This mirrors the QuadcopterEnv.pos_from_vel signature and behavior.

        Arguments:
            - S_t : current state torch.tensor (4,) -> [x, theta, x_dot, th_dot]
            - vel_t_dt : (unused here) next state's velocity torch.tensor (2,)

        Returns:
            - x_t_dt : next state's position torch.tensor (2,) -> [x_next, theta_next]
        """
        x_t_dt = S_t[:2].clone()  # [x, theta]
        x_dot  = S_t[2]
        th_dot = S_t[3]

        x_t_dt += self.dt * torch.FloatTensor([x_dot, th_dot]).to(S_t.device)

        # Optional: wrap angle into [-pi, pi] to keep things tidy
        x_next, th_next = x_t_dt[0], x_t_dt[1]
        th_next = ((th_next + np.pi) % (2 * np.pi)) - np.pi
        return torch.stack([x_next, torch.tensor(th_next, device=S_t.device, dtype=x_t_dt.dtype)])

    # -----------------------
    # Plotting functions
    # -----------------------
    def plot_traj(self, Traj, title: str = ""):
        """Plots the x-theta trajectory (or any desired projection) via external helper."""
        plot_traj(self, Traj, title)

    def traj_comparison(self,
                        traj_1, label_1,
                        traj_2, label_2,
                        title: str = "",
                        traj_3=None, label_3=None,
                        traj_4=None, label_4=None,
                        legend_loc='best'):
        """
        Compares up to 4 trajectories (shape (H, 4)).

        Arguments:
            - traj_1 : first trajectory of shape (H, 4)
            - label_1 : corresponding label to display
            - traj_2 : second trajectory of shape (H, 4)
            - label_2 : corresponding label to display
            - title: optional title of the plot
            - traj_3 : optional third trajectory of shape (H, 4)
            - label_3 : optional corresponding label to display
            - traj_4 : optional fourth trajectory of shape (H, 4)
            - label_4 : optional corresponding label to display
            - legend_loc : optional location of the legend
        """
        traj_comparison(self, traj_1, label_1, traj_2, label_2, title,
                        traj_3, label_3, traj_4, label_4, legend_loc)
