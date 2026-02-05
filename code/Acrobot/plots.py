# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:05:00 2024

@author: Vivek Sharma

Acrobot specific plotting functions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat


def plot_traj(env, Traj, title=""):
    """Plot key components of an Acrobot trajectory"""

    assert len(Traj.shape) == 2, "Trajectory must be a 2D array"
    assert Traj.shape[1] == env.state_size, "Trajectory must contain the full state"
    T = Traj.shape[0]
    time = np.arange(T) * env.dt

    # 1️⃣ Joint angles
    fig, ax = nice_plot()
    if title:
        plt.title(title)
    plt.plot(time, np.rad2deg(Traj[:, 0]), label=r"Joint 1 angle $q_1$ [deg]", linewidth=3)
    plt.plot(time, np.rad2deg(Traj[:, 1]), label=r"Joint 2 angle $q_2$ [deg]", linewidth=3)
    plt.xlabel("Time [s]")
    plt.ylabel("Angles [deg]")
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
    plt.grid(True, alpha=0.3)
    plt.show()

    # 2️⃣ Joint velocities
    fig, ax = nice_plot()
    if title:
        plt.title(title)
    plt.plot(time, np.rad2deg(Traj[:, 2]), label=r"$\dot{q}_1$ [deg/s]", linewidth=3)
    plt.plot(time, np.rad2deg(Traj[:, 3]), label=r"$\dot{q}_2$ [deg/s]", linewidth=3)
    plt.xlabel("Time [s]")
    plt.ylabel("Velocities [deg/s]")
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
    plt.grid(True, alpha=0.3)
    plt.show()

    # 3️⃣ Phase plot (q1 vs q1_dot)
    fig, ax = nice_plot()
    if title:
        plt.title(title + " – Phase portrait (q1)")
    plt.plot(np.rad2deg(Traj[:, 0]), np.rad2deg(Traj[:, 2]), linewidth=3)
    plt.xlabel(r"$q_1$ [deg]")
    plt.ylabel(r"$\dot{q}_1$ [deg/s]")
    plt.grid(True, alpha=0.3)
    plt.show()

    # 4️⃣ Phase plot (q2 vs q2_dot)
    fig, ax = nice_plot()
    if title:
        plt.title(title + " – Phase portrait (q2)")
    plt.plot(np.rad2deg(Traj[:, 1]), np.rad2deg(Traj[:, 3]), linewidth=3)
    plt.xlabel(r"$q_2$ [deg]")
    plt.ylabel(r"$\dot{q}_2$ [deg/s]")
    plt.grid(True, alpha=0.3)
    plt.show()

    # 5️⃣ Configuration trajectory (optional visualization)
    # Simple schematic: tip position in 2D plane
    fig, ax = nice_plot()
    if title:
        plt.title(title + " – End-effector path")
    l1, l2 = env.l1, env.l2
    x_tip = l1 * np.sin(Traj[:, 0]) + l2 * np.sin(Traj[:, 0] + Traj[:, 1])
    y_tip = -l1 * np.cos(Traj[:, 0]) - l2 * np.cos(Traj[:, 0] + Traj[:, 1])
    plt.plot(x_tip, y_tip, linewidth=3)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.show()


def traj_comparison(env, traj_1, label_1, traj_2, label_2, title="",
                    traj_3=None, label_3=None, traj_4=None, label_4=None,
                    legend_loc='best'):
    """Compare multiple Acrobot trajectories in (q1–q2) space."""

    assert len(traj_1.shape) == 2, "Trajectory 1 must be a 2D array"
    assert len(traj_2.shape) == 2, "Trajectory 2 must be a 2D array"
    if traj_3 is not None:
        assert len(traj_3.shape) == 2, "Trajectory 3 must be a 2D array"
    if traj_4 is not None:
        assert len(traj_4.shape) == 2, "Trajectory 4 must be a 2D array"

    fig, ax = nice_plot()
    if title:
        plt.title(title)
    plt.plot(np.rad2deg(traj_1[:, 0]), np.rad2deg(traj_1[:, 1]), label=label_1, linewidth=3)
    plt.plot(np.rad2deg(traj_2[:, 0]), np.rad2deg(traj_2[:, 1]), label=label_2, linewidth=3)
    if traj_3 is not None:
        plt.plot(np.rad2deg(traj_3[:, 0]), np.rad2deg(traj_3[:, 1]), label=label_3, linewidth=3)
    if traj_4 is not None:
        plt.plot(np.rad2deg(traj_4[:, 0]), np.rad2deg(traj_4[:, 1]), label=label_4, linewidth=3)
    plt.xlabel(r"$q_1$ [deg]")
    plt.ylabel(r"$q_2$ [deg]")
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
