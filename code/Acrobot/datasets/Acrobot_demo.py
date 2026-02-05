##https://github.com/tsitsimis/underactuated/tree/master


import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "dataset")
os.makedirs(dataset_dir, exist_ok=True)


class Acrobot:
    """
    Acrobot dynamical system

    Simulates an acrobot based on rigid body manipulator equations.
    The B matrix (mapping from control to angles second derivative)
    is 2x2 meaning that in general case the system can be fully-actuated.
    To model the underactuated acrobot, the first term of control input
    should be zero.
    """

    def __init__(self, m1: float, m2: float, l1: float, l2: float, gravity: float, x0: np.ndarray, u=None,
                 underactuated=True):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.gravity = gravity
        self.underactuated = underactuated

        if u is None:
            if underactuated:
                u = lambda t, x: np.array([[0]])
            else:
                u = lambda t, x: np.array([[0], [0]])
        self.u = u

        self.lc1 = l1 / 2
        self.lc2 = l2 / 2
        self.I1 = m1 * l1 ** 2
        self.I2 = m2 * l2 ** 2

        self.x = x0
        self.t = 0
        self.dt = 0.01

    def get_manipulator_matrices(self, x: np.ndarray):
        """
        Calculates the manipulator matrices M, C, tau_g and B
        Also returns the inverse of M
        """

        q1, q2, q1_dot, q2_dot = x

        m1, m2, I1, I2, l1, l2, lc1, lc2 = self.m1, self.m2, self.I1, self.I2, self.l1, self.l2, self.lc1, self.lc2
        g = self.gravity

        M = np.array([
            [I1 + I2 + m2 * l1 ** 2 + 2 * m2 * l1 * lc2 * np.cos(q2), I2 + m2 * l1 * lc2 * np.cos(q2)],
            [I2 + m2 * l1 * lc2 * np.cos(q2), I2]
        ])

        M_inv = np.array([
            [M[1, 1], -M[0, 1]],
            [-M[1, 0], M[0, 0]]
        ]) / (M[0, 0]*M[1, 1] - M[0, 1]*M[1, 0])

        C = np.array([
            [-2 * m2 * l1 * lc2 * np.sin(q2) * q2_dot, -m2 * l1 * lc2 * np.sin(q2) * q2_dot],
            [m2 * l1 * lc2 * np.sin(q2) * q1_dot, 0]
        ])

        tau_g = np.array([
            [-m1 * g * lc1 * np.sin(q1) - m2 * g * (l1 * np.sin(q1) + lc2 * np.sin(q1 + q2))],
            [-m2 * g * lc2 * np.sin(q1 + q2)]
        ])

        if self.underactuated:
            B = np.array([
                [0],
                [1]
            ])
        else:
            B = np.eye(2)

        return M, M_inv, C, tau_g, B

    def dynamics(self, t: float, x: np.ndarray):
        """
        Implements the system's differential equations and returns
        state derivatives given current time and state
        """

        q_dot = np.array([[x[2]], [x[3]]])

        M, M_inv, C, tau_g, B = self.get_manipulator_matrices(x)

        x34_dot = M_inv.dot(tau_g + B.dot(self.u(t, x)) - C.dot(q_dot))
        x1_dot = x[2]
        x2_dot = x[3]
        return np.array([x1_dot, x2_dot, x34_dot[0][0], x34_dot[1][0]])

    def step(self):
        """
        Applies numerical integration in system dynamics and updates
        current system's state
        """

        sol = solve_ivp(self.dynamics, [self.t, self.t + self.dt], self.x)

        self.x = sol.y[:, -1]
        self.t += self.dt

    def playback(self, fig, ax, T, save=False, show_time=True, return_data=True):
        """
        Simulates the system until time T and animates it in a matplotlib figure.

        Returns (optionally) a dictionary containing:
            - 'time': array of time steps
            - 'states': array of shape (n, state_dim)
            - 'actions': array of shape (n, action_dim)
            - 'actions_norm': actions normalized to [-1, 1]
        """

        # ---- Simulation setup ----
        time_steps = np.arange(0, T, self.dt)
        n = len(time_steps)
        state_dim = self.x.shape[0]
        if callable(self.u):
            # evaluate once at current state
            u_sample = np.atleast_1d(self.u(0, self.x))
            act_dim = u_sample.size
        else:
            act_dim = np.atleast_1d(self.u).size
        #act_dim = self.u.shape[0] if hasattr(self, "u") else 1
        print(act_dim)

        states_cache = np.zeros((n, state_dim))
        actions_cache = np.zeros((n, act_dim))


        for i, t in enumerate(time_steps):
            #u_t = np.atleast_1d(self.u(t, self.x))
            u_t = np.asarray(self.u(t, self.x)).squeeze()
            if u_t.ndim == 0:
                u_t = np.array([u_t])  # handle scalar control
            actions_cache[i, :] = u_t
            actions_cache[i, :] = u_t
            sol = solve_ivp(self.dynamics, [self.t, self.t + self.dt], self.x)
            self.x = sol.y[:, -1]
            self.t += self.dt
            states_cache[i, :] = self.x

        # ---- Normalize actions to [-1, 1] ----
        u_min = np.min(actions_cache, axis=0)
        u_max = np.max(actions_cache, axis=0)
        # Avoid division by zero for constant control channels
        denom = (u_max - u_min + 1e-8)
        u_norm = 2.0 * (actions_cache - u_min) / denom - 1.0

        # ---- Animation setup ----
        plt.ion()
        plt.axis("off")
        ax.axis("equal")
        ax.set_xlim(-1.1 * (self.l1 + self.l2), 1.1 * (self.l1 + self.l2))
        ax.scatter([0], [0], marker="o", c="k")
        ax.plot([-self.l1 / 2, self.l1 / 2], [0, 0], c="k", linestyle="--")
        p1, = ax.plot([], [], c="k")
        p2, = ax.plot([], [], c="k")
        middle_joint_p = ax.scatter([], [], c="b", zorder=10)

        if show_time:
            time_text = ax.text(x=0, y=1.2 * (self.l1 + self.l2), s="t=0")
        plt.show()

        # ---- Animation loop ----
        for i, t in enumerate(time_steps):
            if i % 10 != 0:
                continue

            th1, th2 = states_cache[i, 0], states_cache[i, 1]
            x1, y1 = self.l1 * np.sin(th1), -self.l1 * np.cos(th1)
            x2, y2 = x1 + self.l2 * np.sin(th1 + th2), y1 - self.l2 * np.cos(th1 + th2)

            p1.set_data([0, x1], [0, y1])
            p2.set_data([x1, x2], [y1, y2])
            middle_joint_p.set_offsets([[x1, y1]])

            if show_time:
                time_text.set_text(f"t={t:.2f}s")

            fig.canvas.draw()
            plt.pause(self.dt)


        # ---- Return results ----
        if return_data:
            data = {
                "time": time_steps,
                "states": states_cache,
                "actions": actions_cache,
                "actions_norm": u_norm,
            }
            return data



class SystemLinearizer:
    def __init__(self, system, x0, tau_g_dq, B_dq=None):
        self.system = system
        self.x0 = x0
        self.tau_g_dq = tau_g_dq

        M, M_inv, C, tau_g, B = system.get_manipulator_matrices(x0)
        n_half = M.shape[0]

        if B_dq is None:
            B_dq = np.zeros((n_half, 1))
        self.B_dq = B_dq

        A_lin_top = np.concatenate((
            np.zeros((n_half, n_half)),
            np.eye(n_half)
        ), axis=1)
        A_lin_bottom = np.concatenate((
            M_inv.dot(tau_g_dq),
            -M_inv.dot(C)
        ), axis=1)
        self.A_lin = np.concatenate((A_lin_top, A_lin_bottom), axis=0)

        self.B_lin = np.concatenate((
            np.zeros((n_half, B.shape[1])),
            M_inv.dot(B)
        ), axis=0)



class FLC:
    def __init__(self, plant, v):
        self.plant = plant
        self.v = v

    def controller(self, t, x):
        n = x.shape[0]
        q_dot = x[int(n/2):][:, None]

        M, M_inv, C, tau_g, B = self.plant.get_manipulator_matrices(x)
        u = np.linalg.inv(B).dot(M.dot(self.v(t, x)) + C.dot(q_dot) - tau_g)
        return u


class LQR:
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        self.S = solve_continuous_are(A, B, Q, R)

    def controller(self, t, x):
        return -np.linalg.inv(self.R).dot(self.B.T).dot(self.S).dot(x[:, None])



if __name__ == "__main__":

    controller_='LQR'
    system_name = "Acrobot"

    if controller_ == 'LQR':
        x_goal = np.array([np.pi, 0, 0, 0])
        #for i in range(10):
            # Plant
        plant = Acrobot(m1=1, m2=2, l1=1, l2=2, gravity=10, x0=np.array([np.pi * 0.9, 0, 0, 0]), underactuated=True)
        #  (b) :LQR

        m1, m2, l1, l2, lc1, lc2, g = plant.m1, plant.m2, plant.l1, plant.l2, plant.lc1, plant.lc2, plant.gravity
        tau_g_dq = np.array([
            [g * (m1 * lc1 + m2 * l1 + m2 * lc2), m2 * g * lc2],
            [m2 * g * lc2, m2 * g * lc2]
        ])
        sl = SystemLinearizer(plant, x0=x_goal, tau_g_dq=tau_g_dq)

        # # LQR controller
        # lqr = LQR(A=sl.A_lin, B=sl.B_lin, Q=np.eye(4), R=np.eye(1))
        # plant.u = lambda t, x: lqr.controller(t, x - x_goal)
        #
        # # Animate
        # # fig, ax = plt.subplots(figsize=(5, 5))
        # # plant.playback(fig=fig, ax=ax, T=10, save=True)
        # fig, ax = plt.subplots(figsize=(5, 5))
        # traj = plant.playback(fig=fig, ax=ax, T=10, save=True)
        #
        # print("States:", traj["states"].shape)
        # print("Raw actions:", traj["actions"][:5])
        # print("Normalized actions [-1,1]:", traj["actions_norm"][:5])

        # LQR controller setup
        lqr = LQR(A=sl.A_lin, B=sl.B_lin, Q=np.eye(4), R=np.eye(1))
        plant.u = lambda t, x: lqr.controller(t, x - x_goal)

        # Simulation parameters
        T = 10
        num_conditions = 10  # number of random initial conditions
        state_dim = plant.x.shape[0]
        # sample initial conditions around some nominal value (example)
        n_values = np.linspace(0.1, 0.9, num_conditions)

        # Build list of initial states
        x0_list = [np.array([n * np.pi, 0.0, 0.0, 0.0]) for n in n_values]
        #x0_list = [np.random.uniform(-0.2, 0.2, size=state_dim) for _ in range(num_conditions)]

        # Preallocate placeholders (they’ll be stacked)
        Trajs_list, Actions_list = [], []

        # Run simulations for each initial condition
        for i, x0 in enumerate(x0_list):
            print(f"Simulating trajectory {i + 1}/{num_conditions} with x0 = {x0}")

            # Reset plant to new initial condition
            plant.x = x0.copy()
            plant.t = 0.0

            # Run simulation (no animation for dataset generation)
            fig, ax = plt.subplots(figsize=(5, 5))
            traj = plant.playback(fig=fig, ax=ax, T=T, save=False, show_time=False, return_data=True)
            #plt.close(fig)

            # Store results
            Trajs_list.append(traj["states"])  # shape (steps, n)
            Actions_list.append(traj["actions_norm"])  # shape (steps, m)

        # Stack into arrays
        Trajs = np.stack(Trajs_list, axis=0)  # shape (N_conditions, steps, n)
        Actions = np.stack(Actions_list, axis=0)  # shape (N_conditions, steps, m)

        # Build dataset dictionary
        dataset = {"Trajs": Trajs, "Actions": Actions}

        # Print summary
        print(f"dataset['Trajs'] shape:   {dataset['Trajs'].shape}")
        print(f"dataset['Actions'] shape: {dataset['Actions'].shape}")
        num_trajs = dataset["Trajs"].shape[0]
        num_steps = dataset["Trajs"].shape[1]
        filename = f"{system_name}_{num_trajs}trajs_{num_steps}steps.npz"
        save_path = os.path.join(dataset_dir, filename)

        # ---- Save ----
        np.savez(save_path, **dataset)
        print(f"✅ Saved dataset → {save_path}")

        # np.savez("datasets/acrobot.npz", **dataset)
        # print("Saved dataset → playback_dataset.npz")

    elif controller_ == 'FLC':
        #  (b) :Feedback Linearization Controller
        #     # Make system feedback-equivalent to a linear system
        #     # controlled with a simple PD controller
        plant = Acrobot(m1=1, m2=1, l1=1, l2=1, gravity=10, x0=np.array([np.pi / 4, np.pi / 4, 0, 0]), underactuated=False)
        plant.u = FLC(plant, lambda t, x: np.array([[-(x[0] - np.pi) - x[2]], [-(x[1] - 0) - x[3]]])).controller

        # Animate
        fig, ax = plt.subplots(figsize=(5, 5))
        traj = plant.playback(fig=fig, ax=ax, T=10, save=True)

        print("States:", traj["states"].shape)
        print("Raw actions:", traj["actions"][:5])
        print("Normalized actions [-1,1]:", traj["actions_norm"][:5])


