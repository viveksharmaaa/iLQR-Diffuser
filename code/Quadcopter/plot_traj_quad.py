import numpy as np
import matplotlib.pyplot as plt

# ---- Load dataset ----
data = np.load("datasets/Quadcopter_1000trajs_200steps.npz")
Trajs = data["Trajs"]  # shape (4, 200, 17)
Actions = data["Actions"]

num_trajs, steps, n_states_ = Trajs.shape
m = Actions.shape[2]
time = np.arange(steps)

n_states = 3 #Choose the states for plotting
# ---- Create figure with 17 subplots ----
fig, axs = plt.subplots(n_states, 1, figsize=(8, 8), sharex=True)

for j in range(n_states): #n_states
    for i in range(num_trajs):
        axs[j].plot(time, Trajs[i, :, j], lw=1.2, alpha=0.8)
    axs[j].set_ylabel(f"x{j+1}")
    axs[j].grid(alpha=0.3)

axs[-1].set_xlabel("Time step")
fig.suptitle("All 17 State Trajectories (4 rollouts)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# ============================================================
# Figure 2: Control Inputs
# ============================================================
fig_u, axs_u = plt.subplots(m, 1, figsize=(10, 3 * m), sharex=True)
if m == 1:
    axs_u = [axs_u]  # make iterable if single control input

for k in range(m):
    for i in range(num_trajs):
        axs_u[k].plot(time, Actions[i, :, k], lw=1.2, alpha=0.8)
    axs_u[k].set_ylabel(fr"$u_{k+1}$")
    axs_u[k].grid(alpha=0.3)
    axs_u[k].set_title(f"Control Input $u_{k+1}$", fontsize=11)

axs_u[-1].set_xlabel("Time step")
fig_u.suptitle("Control Inputs (4 rollouts)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()