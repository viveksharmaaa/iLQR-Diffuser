import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams.update({
    "font.size": 20,         # default text size
    "axes.titlesize": 20,    # title
    "axes.labelsize": 20,    # x and y labels
    "xtick.labelsize": 12,   # x tick labels
    "ytick.labelsize": 12,   # y tick labels
    "legend.fontsize": 20,   # legend
    "font.weight": "bold",
})

# ---- Load dataset ----
data = np.load("Cartpole_10trajs_1200steps.npz")
Trajs = data["Trajs"]     # shape (10, 1000, 4)
Actions = data["Actions"] # shape (10, 1000, 1) or (10, 1000, m)

num_trajs, steps, n = Trajs.shape
m = Actions.shape[2]
time = np.arange(steps)


# ---- Plot ----
fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

state_labels = [r"$x$ [m]", r"$\theta$ [rad]",
                r"$\dot{x}$ [m/s]", r"$\dot{\theta}$ [rad/s]"]

for j in range(n):
    for i in range(num_trajs):
        axs[j].plot(time, Trajs[i, :, j], lw=1, alpha=0.8)
    axs[j].set_ylabel(state_labels[j])
    axs[j].grid(alpha=0.3)

axs[-1].set_xlabel("Time step")
fig.suptitle("Acrobot Trajectories (All 4 States)")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

fig_u, ax_u = plt.subplots(figsize=(8, 3))
for k in range(m):
    for i in range(num_trajs):
        ax_u.plot(time, Actions[i, :, k], lw=1.2, alpha=0.8)
    ax_u.set_ylabel(fr"$u_{k+1}$")
    ax_u.grid(alpha=0.3)

ax_u.set_xlabel("Time step")
ax_u.set_title("Control Inputs over Time")
plt.tight_layout()

plt.show()
