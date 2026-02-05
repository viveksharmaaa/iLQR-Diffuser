import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import imageio
from scipy.linalg import solve_continuous_are
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "datasets")
os.makedirs(dataset_dir, exist_ok=True)



class CartPole:
    """
    Cart-Pole dynamical system
    """

    def __init__(self, m_c: float, m_p: float, l: float, gravity: float, x0: np.ndarray, u=None, underactuated=True):
        self.m_c = m_c
        self.m_p = m_p
        self.l = l
        self.gravity = gravity
        self.underactuated = underactuated

        if u is None:
            if underactuated:
                u = lambda t, x: np.array([[0]])
            else:
                u = lambda t, x: np.array([[0], [0]])
        self.u = u

        self.x = x0
        self.t = 0
        self.dt = 0.01

    def get_manipulator_matrices(self, x: np.ndarray):
        """
        Calculates the manipulator matrices M, C, tau_g and B
        Also returns the inverse of M
        """

        d, theta, d_dot, theta_dot = x

        m_c, m_p, l = self.m_c, self.m_p, self.l
        g = self.gravity

        M = np.array([
            [m_c + m_p, m_p * l * np.cos(theta)],
            [m_p * l * np.cos(theta), m_p * l ** 2]
        ])

        M_inv = np.array([
            [M[1, 1], -M[0, 1]],
            [-M[1, 0], M[0, 0]]
        ]) / (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])

        C = np.array([
            [0, -m_p * l * theta_dot * np.sin(theta)],
            [0, 0]
        ])

        tau_g = np.array([
            [0],
            [-m_p * g * l * np.sin(theta)]
        ])

        if self.underactuated:
            B = np.array([
                [1],
                [0]
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

    def playback(self, fig, ax, T, save=False, show_time=True, callback=None,return_data=True):
        """
        Simulates the system until time T and animates it in
        a matplotlib figure
        """

        time_steps = np.arange(0, T, self.dt)
        n = time_steps.shape[0]
        state_dim = self.x.shape[0]
        if callable(self.u):
            # evaluate once at current state
            u_sample = np.atleast_1d(self.u(0, self.x))
            act_dim = u_sample.size
        else:
            act_dim = np.atleast_1d(self.u).size
        #act_dim = self.u.shape[0] if hasattr(self, "u") else 1
        print(act_dim)

        states_cache = np.zeros((n, 4))
        actions_cache = np.zeros((n, act_dim))
        for i, t in enumerate(time_steps):
            self.step()
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

        frames = [None] * n

        plt.ion()
        plt.axis("off")
        ax.axis("equal")
        ax.set_xlim(-2 * self.l, 2 * self.l)

        ax.plot([-2 * self.l, 2 * self.l], [0, 0], c="k", linestyle="--")
        p_cart_top, = ax.plot([], [], c="k")
        p_cart_bottom, = ax.plot([], [], c="k")
        p_cart_left_side, = ax.plot([], [], c="k")
        p_cart_right_side, = ax.plot([], [], c="k")
        p_pole, = ax.plot([], [], c="k")
        pole_joint_p = ax.scatter([], [], c="k", zorder=10)

        if show_time:
            time_text = ax.text(x=0, y=2 * self.l, s="t=0")
        plt.show()

        for i, t in enumerate(time_steps):
            if i % 10 != 0:
                continue
            p_cart_top.set_data([states_cache[i, 0] - self.l / 4, states_cache[i, 0] + self.l / 4],
                                [self.l / 5, self.l / 5])
            p_cart_bottom.set_data([states_cache[i, 0] - self.l / 4, states_cache[i, 0] + self.l / 4], [0, 0])
            p_cart_left_side.set_data([states_cache[i, 0] - self.l / 4, states_cache[i, 0] - self.l / 4],
                                      [0, self.l / 5])
            p_cart_right_side.set_data([states_cache[i, 0] + self.l / 4, states_cache[i, 0] + self.l / 4],
                                       [0, self.l / 5])

            p_pole.set_data([states_cache[i, 0], states_cache[i, 0] + self.l * np.sin(states_cache[i, 1])],
                            [self.l / 10, -self.l * np.cos(states_cache[i, 1])])
            pole_joint_p.set_offsets([[states_cache[i, 0], self.l / 10]])

            if show_time:
                time_text.set_text(f"t={np.round(t, 1)}")

            if callback is not None:
                callback(ax, i, t, states_cache)

            fig.canvas.draw()

            plt.pause(self.dt)

        #     if save:
        #         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        #         frames[i] = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #
        # if save:
        #     imageio.mimsave(f"./cartpole_{int(time.time())}.gif", [f for f in frames if f is not None], fps=15)

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

    controller ='energy'
    system_name='Cartpole'
    if controller == 'LQR':
        # Plant
        plant = CartPole(m_c=1, m_p=1, l=1, gravity=10, x0=np.array([-2, np.pi*0, 0, 0]), underactuated=True)

        # Linearize around fixed point (vertical position, zero velocity)
        x_goal = np.array([0, np.pi, 0, 0])
        m_p, l, g = plant.m_p, plant.l, plant.gravity
        tau_g_dq = np.array([
            [0, 0],
            [0, m_p * l * g]
        ])
        sl = SystemLinearizer(plant, x0=x_goal, tau_g_dq=tau_g_dq)

        # LQR controller
        lqr = LQR(A=sl.A_lin, B=sl.B_lin, Q=10*np.eye(4), R=np.eye(1))
        plant.u = lambda t, x: lqr.controller(t, x - x_goal)

        # Animate
        fig, ax = plt.subplots(figsize=(5, 5))
        plant.playback(fig=fig, ax=ax, T=10, save=True)

    elif controller == 'FLC':
        # Plant
        plant = CartPole(m_c=1, m_p=1, l=1, gravity=10, x0=np.array([-1, 0, 0, 0]), underactuated=False)

        # Feedback Linearization Controller
        # Make system feedback-equivalent to a linear system
        # controlled with a simple PD controller
        plant.u = FLC(plant, lambda t, x: np.array([[-(x[0] - 0) - x[2]], [-(x[1] - np.pi) - x[3]]])).controller

        # Animate
        fig, ax = plt.subplots(figsize=(5, 5))
        plant.playback(fig=fig, ax=ax, T=10, save=True)

    elif controller == 'energy':
        T = 12
        Trajs_list, Actions_list = [], []
        for i in range(10):
            plant = CartPole(m_c=1, m_p=1, l=1, gravity=10, x0=np.array([-1, np.pi / (i+1), 0, 0]), underactuated=True)
            # Energy shaping controller
            m_c, m_p, l, g = plant.m_c, plant.m_p, plant.l, plant.gravity
            k_e, k_p, k_d = 5, 10, 10  # gains after experimentation


            def energy_shaping_controller(t, x):
                d, theta, d_dot, theta_dot = x

                T = (1 / 2) * m_p * l ** 2 * theta_dot ** 2
                U = -m_p * g * l * np.cos(theta)
                E = T + U
                E_d = m_p * g * l

                u_energy_shaping = k_e * theta_dot * np.cos(theta) * (E - E_d)

                # Add extra terms to bring the cart in zero position
                u_desired = u_energy_shaping - k_p * d - k_d * d_dot

                # Apply Partial Feedback Linearization (PFL)
                u = (m_c + m_p - m_p * np.cos(theta) ** 2) * u_desired - (
                        m_p * g * np.sin(theta) * np.cos(theta) + m_p * l * np.sin(theta) * theta_dot ** 2)

                return u


            # LQR controller of linearized system
            x_goal = np.array([0, np.pi, 0, 0])
            sl = SystemLinearizer(plant, x0=x_goal, tau_g_dq=np.array([[0, 0], [0, m_p * g * l]]))
            x_init = np.array([-1, np.pi / (i+1), 0, 0])

            lqr = LQR(A=sl.A_lin, B=sl.B_lin, Q=10 * np.eye(4), R=np.eye(1))


            def lqr_controller(t, x):
                return lqr.controller(t, x - x_goal)


            # Mixed controller
            def mixed_controller(t, x):
                if (abs(x[0]) > 0.1) or (abs(x[1] - np.pi) > np.pi / 30):
                    return energy_shaping_controller(t, x)
                return lqr_controller(t, x)


            plant.u = mixed_controller

            # Animate
            fig, ax = plt.subplots(figsize=(5, 5))
            controller_text = ax.text(x=-2 * plant.l, y=2 * plant.l, s="", color="r", size=12)


            def animation_callback(ax, i, t, states_cache):
                x = states_cache[i, :]
                if (abs(x[0]) > 0.1) or (abs(x[1] - np.pi) > np.pi / 30):
                    controller_text.set_text("Energy Shaping")
                else:
                    controller_text.set_text("LQR")

            print(f"Simulating trajectory {i + 1} with x0 = {x_init}")

            # Reset plant to new initial condition
            plant.x = x_init.copy()
            plant.t = 0.0

            # Run simulation (no animation for dataset generation)
            fig, ax = plt.subplots(figsize=(5, 5))
            traj = plant.playback(fig=fig, ax=ax, T=T, save=False, show_time=False, return_data=True)
            plt.close(fig)

            # Store results
            Trajs_list.append(traj["states"])  # shape (steps, n)
            Actions_list.append(traj["actions_norm"])  # shape (steps, m)
            plant.playback(fig=fig, ax=ax, T=12, save=True, callback=animation_callback)

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