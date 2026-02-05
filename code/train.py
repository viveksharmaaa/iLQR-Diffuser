# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:01:12 2024

@author: Jean-Baptiste Bouvier

Main script to train a Diffusion Transfomer for a given environment
"""

# import sys; print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['/home/sharma/Projects/DDAT/code'])
# sys.path.extend(['/home/sharma/Projects/DDAT/code/Cartpole/datasets'])

import torch

from utils.utils import set_seed
from DiT.ODE import ODE
from utils.loaders import make_env, load_datasets, load_proj
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "palatino"
plt.rcParams.update({
    "font.size": 12,         # default text size
    "axes.titlesize": 16,    # title
    "axes.labelsize": 14,    # x and y labels
    "xtick.labelsize": 12,   # x tick labels
    "ytick.labelsize": 12,   # y tick labels
    "legend.fontsize": 12,   # legend
})


#%% Hyperparameters

modality = "S" # whether diffusion predicts only states "S", states and actions "SA", or only actions "A"
env_name = "Quadcopter" # name of the environment in ["Hopper", "Walker", "HalfCheetah", "Quadcopter", "GO1", "GO2","Cartpole","Acrobot"]
proj_name = "Adm" # name of the projector in [None, "Ref", "Adm", "SA", "A"]
conditioning = "s0" # attributes on which the diffusion model is conditioned in [None, "s0", "cmd", "s0_cmd"]

extra_name = "" # string to add to the diffusion model to differentiate it from others
n_gradient_steps = 11
if env_name == "Cartpole":
    batch_size = 12 #should be number of trajectories 10 for no proj, 12 for Ref Projection
    n_gradient_steps = 1500
    model_size_d_model = 256
elif env_name == "Acrobot":
    batch_size = 32
else:
    batch_size = 256

time_limit = None # stops the training after this many seconds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(0)

if env_name == "Cartpole":
    N_trajs=10
    H =1200
elif env_name == "Acrobot":
    N_trajs=10
    H = 1000
else:
    N_trajs = 1000  # number of trajectories in the dataset 1000
    H = 200  # horizon, length of each trajectory in the dataset 200


#%% Default environment parameters

print("Device:", device)
env, model_size, H, N_trajs = make_env(env_name, modality)
x, attr, attr_dim = load_datasets(env_name, modality, conditioning, N_trajs, H, device)
proj = load_proj(proj_name, env, device, modality, dataset=x)


#%% Default DiT parameters
ode = ODE(env, modality=modality, attr_dim=attr_dim, device=device, **model_size,
          projector=None)

if proj_name is not None: # start training from the base model
    assert ode.load(extra=extra_name), "Train without projections first"
    ode.update_projector(proj)


# #Plot all trajectories
x_np = x.cpu().numpy()
t = np.arange(x_np.shape[1])
if env_name=='Quadcopter':
    labels = [r"$x$", r"$y$"]
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    for i in range(x_np.shape[0]):
        plt.plot(x_np[i, :, 0], x_np[i, :, 1], alpha=0.7, label=f"Traj {i + 1}")
    # plt.Circle((2.5, 0.5), 0.7, color='r')
    # plt.Circle((5.5, -0.5), 0.7, color='red')
    ax.add_patch(plt.Circle((2.5, 0.5), 0.7, color='r'))
    ax.add_patch(plt.Circle((5.5, -0.5), 0.7, color='red'))
    ax.axis("equal")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title("Position x vs y plot")
    plt.show()

elif env_name=='Cartpole':
    labels = [r"$x$", r"$\theta$", r"$\dot{x}$", r"$\dot{\theta}$"]
    plt.figure(figsize=(6, 5))
    for i in range(x_np.shape[0]):
        plt.plot(x_np[i, :, 0], x_np[i, :, 1], alpha=0.7, label=f"Traj {i + 1}")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\theta$")
    plt.title("Phase plot")

    fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
    for i in range(4):
        axs[i].plot(t, x_np[:, :, i].T, alpha=0.7)
        axs[i].set_ylabel(labels[i])
    axs[-1].set_xlabel("Time step")
    plt.tight_layout();
    plt.show()

ode.train(x, attributes=attr, batch_size=batch_size, n_gradient_steps=n_gradient_steps,extra=extra_name, time_limit=time_limit)



