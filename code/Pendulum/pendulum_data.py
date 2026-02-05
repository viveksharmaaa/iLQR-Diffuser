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


class Pendulum:
    """
    Simple pendulum dynamical system

    Simulates a pendulum based on rigid body manipulator equations.
    Accepts an input control function that depends on current time
    and state.
    """

    def __init__(self, mass: float, length: float, friction: float, gravity: float, x0: np.ndarray, u=None):
        self.mass = mass
        self.length = length
        self.friction = friction
        self.gravity = gravity

        if u is None:
            u = lambda t, x: np.array([[0]])
        self.u = u

        self.x = x0
        self.t = 0
        self.dt = 0.01

    def get_manipulator_matrices(self, x):
        """
        Calculates the manipulator matrices M, C, tau_g and B
        Also returns the inverse of M
        """

        q, q_dot = x

        m, l, b, g = self.mass, self.length, self.friction, self.gravity

        M = np.array([[m * l ** 2]])
        M_inv = np.array([[1 / (m * l ** 2)]])

        C = np.array([[b]])

        tau_g = np.array([[- m * g * l * np.sin(q)]])

        B = np.array([[1]])

        return M, M_inv, C, tau_g, B

    def dynamics(self, t, x):
        """
        Implements the system's differential equations and returns
        state derivatives given current time and state
        """

        q_dot = np.array([[x[1]]])

        M, M_inv, C, tau_g, B = self.get_manipulator_matrices(x)
        x2_dot = M_inv.dot(tau_g + B.dot(self.u(t, x)) - C.dot(q_dot))
        x2_d= x2_dot[0][0]


        return np.array([x[1], x2_d])

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
        #states_cache = np.zeros((n, 2))
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
            self.step()
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


        frames = [None] * n

        plt.ion()
        plt.axis("off")
        ax.axis("equal")
        #ax.set_xlim(-2 * self.length, 2 * self.length)
        #ax.set_ylim(-1.1 * self.length, 1.1 * self.length)

        ax.scatter([0], [0], marker="o", c="b")
        ax.plot([-self.length / 2, self.length / 2], [0, 0], c="k", linestyle="--")
        p, = ax.plot([], [], c="k")

        if show_time:
            time_text = ax.text(x=0, y=1.2 * self.length, s="t=0")
        plt.show()

        for i, t in enumerate(time_steps):
            if i % 10 != 0:
                continue
            p.set_data([0, self.length * np.sin(states_cache[i, 0])], [0, -self.length * np.cos(states_cache[i, 0])])
            fig.canvas.draw()

            if show_time:
                time_text.set_text(f"t={np.round(t, 1)}")

            if callback is not None:
                callback(ax, i, t, states_cache)

            plt.pause(self.dt)

        #     if save:
        #         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        #         frames[i] = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #
        # if save:
        #     imageio.mimsave(f"./pendulum_{int(time.time())}.gif", [f for f in frames if f is not None], fps=15)

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
    system_name = "LQR"

    if controller_ == 'LQR':
        plant = Pendulum(mass=1, length=1, friction=0.5, gravity=10, x0=np.array([np.pi* 3/4, 0]))

        x_goal = np.array([np.pi, 0])
        sl = SystemLinearizer(plant, x0=x_goal, tau_g_dq=np.array([[10]]))

        lqr = LQR(sl.A_lin, sl.B_lin, 10*np.eye(2), np.eye(1))
        plant.u = lambda t, x: lqr.controller(t, x - x_goal)

        # Simulation parameters
        T = 2
        num_conditions = 10  # number of random initial conditions
        state_dim = plant.x.shape[0]
        # sample initial conditions around some nominal value (example)
        n_values = np.linspace(-1., 1, num_conditions)

        # Build list of initial states
        x0_list = [np.array([n * np.pi, 0.0]) for n in n_values]
        # x0_list = [np.random.uniform(-0.2, 0.2, size=state_dim) for _ in range(num_conditions)]

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
            # plt.close(fig)

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

        # Animate
        fig, ax = plt.subplots(figsize=(5, 5))
        plant.playback(fig=fig, ax=ax, T=4, save=True)

    elif controller_ == 'FLC':
        # Plant
        plant = Pendulum(mass=1, length=1, friction=0.5, gravity=10, x0=np.array([np.pi / 4, -5]))

        # Feedback Linearization Controller
        # Make system feedback-equivalent to a linear system
        # controlled with a simple PD controller
        plant.u = FLC(plant, lambda t, x: np.array([[-2 * x[1] - 2 * (x[0] - np.pi)]])).controller

        # Animate
        fig, ax = plt.subplots(figsize=(5, 5))
        plant.playback(fig=fig, ax=ax, T=50)

    elif controller_ == 'energy':
        # Plant
        plant = Pendulum(mass=1, length=1, friction=0.5, gravity=10, x0=np.array([np.pi / 6, 0]))

        # Energy shaping controller
        k = 100.0
        m, g, l = plant.mass, plant.gravity, plant.length
        u_max = m * g * l / 4


        def energy_shaping_controller(t, x, underactuated=True):
            u = -k * x[1] * ((-m * g * l * np.cos(x[0]) + (1 / 2) * m * (l ** 2) * (x[1] ** 2)) - m * g * l)
            if underactuated:
                u = np.clip(u, -u_max, u_max)

            return u


        # LQR controller of linearized system
        x_goal = np.array([np.pi, 0])
        sl = SystemLinearizer(plant, x0=x_goal, tau_g_dq=np.array([[10]]))

        lqr = LQR(sl.A_lin, sl.B_lin, 10 * np.eye(2), np.eye(1))


        def lqr_controller(t, x):
            return lqr.controller(t, x - x_goal)


        # Mixed controller
        def mixed_controller(t, x):
            if np.linalg.norm(x - x_goal) > 0.1:
                return energy_shaping_controller(t, x)
            return lqr_controller(t, x)


        plant.u = mixed_controller

        # Animate
        fig, ax = plt.subplots(figsize=(5, 5))
        controller_text = ax.text(x=-1.0 * plant.length, y=1.2 * plant.length, s="", color="r", size=12)


        def animation_callback(ax, i, t, states_cache):
            x = states_cache[i, :]
            if np.linalg.norm(x - x_goal) > 0.1:
                controller_text.set_text("Energy Shaping")
            else:
                controller_text.set_text("LQR")


        plant.playback(fig=fig, ax=ax, T=15, save=True, callback=animation_callback)