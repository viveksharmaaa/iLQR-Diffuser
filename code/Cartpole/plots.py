# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:32:09 2024

@author: Jean-Baptiste Bouvier (style matched)

Cartpole specific plotting functions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat


def plot_traj(env, Traj, title=""):
    """Plot key components of a Cartpole trajectory"""

    assert len(Traj.shape) == 2, "Trajectory must be a 2D array"
    assert Traj.shape[1] == env.state_size, "Trajectory must contain the full state"
    T = Traj.shape[0]
    time = np.arange(T) * env.dt

    # 1️⃣ Time evolution of states
    fig, ax = nice_plot()
    if title:
        plt.title(title)
    plt.plot(time, Traj[:, 0], label="Cart position x [m]", linewidth=3)
    plt.plot(time, np.rad2deg(Traj[:, 1]), label="Pole angle θ [deg]", linewidth=3)
    plt.xlabel("Time [s]")
    plt.ylabel("States")
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
    plt.show()

    # 2️⃣ Velocities
    fig, ax = nice_plot()
    if title:
        plt.title(title)
    plt.plot(time, Traj[:, 2], label="Cart velocity $\dot{x}$ [m/s]", linewidth=3)
    plt.plot(time, np.rad2deg(Traj[:, 3]), label="Pole angular velocity $\dot{θ}$ [deg/s]", linewidth=3)
    plt.xlabel("Time [s]")
    plt.ylabel("Velocities")
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
    plt.show()

    # 3️⃣ Phase plot (angle vs angular velocity)
    fig, ax = nice_plot()
    if title:
        plt.title(title + " – Phase portrait")
    plt.plot(np.rad2deg(Traj[:, 1]), np.rad2deg(Traj[:, 3]), linewidth=3)
    plt.xlabel("θ [deg]")
    plt.ylabel("θ̇ [deg/s]")
    plt.grid(True, alpha=0.3)
    plt.show()

    # 4️⃣ Cart position vs pole angle
    fig, ax = nice_plot()
    if title:
        plt.title(title + " – State trajectory")
    plt.plot(Traj[:, 0], np.rad2deg(Traj[:, 1]), linewidth=3)
    plt.xlabel("x [m]")
    plt.ylabel("θ [deg]")
    plt.grid(True, alpha=0.3)
    plt.show()


def traj_comparison(env, traj_1, label_1, traj_2, label_2, title="",
                    traj_3=None, label_3=None, traj_4=None, label_4=None,
                    legend_loc='best'):
    """Compares given cartpole trajectories (x–θ plane)."""

    assert len(traj_1.shape) == 2, "Trajectory 1 must be a 2D array"
    assert len(traj_2.shape) == 2, "Trajectory 2 must be a 2D array"
    if traj_3 is not None:
        assert len(traj_3.shape) == 2, "Trajectory 3 must be a 2D array"
    if traj_4 is not None:
        assert len(traj_4.shape) == 2, "Trajectory 4 must be a 2D array"

    fig, ax = nice_plot()
    if title:
        plt.title(title)
    plt.plot(traj_1[:, 0], np.rad2deg(traj_1[:, 1]), label=label_1, linewidth=3)
    plt.plot(traj_2[:, 0], np.rad2deg(traj_2[:, 1]), label=label_2, linewidth=3)
    if traj_3 is not None:
        plt.plot(traj_3[:, 0], np.rad2deg(traj_3[:, 1]), label=label_3, linewidth=3)
    if traj_4 is not None:
        plt.plot(traj_4[:, 0], np.rad2deg(traj_4[:, 1]), label=label_4, linewidth=3)
    plt.xlabel("x [m]")
    plt.ylabel("θ [deg]")
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9, loc=legend_loc)
    plt.grid(True, alpha=0.3)
    plt.show()


def nice_plot():
    """Makes the plot nice (white spines, clean fonts)"""
    fig = plt.gcf()
    ax = fig.gca()
    plt.rcParams.update({'font.size': 16})
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')
    return fig, ax
