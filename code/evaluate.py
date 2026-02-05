# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:46:22 2025

@author: Jean-Baptiste Bouvier

Main script to load and evaluate a pre-trained Diffusion Transfomer
"""

# import sys; print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['/home/sharma/Projects/DDAT/code'])
# sys.path.extend(['/home/sharma/Projects/DDAT/code/Cartpole/datasets'])

import torch
import numpy as np


from utils.loaders import make_env, load_proj
from utils.utils import set_seed, open_loop
from utils.inverse_dynamics import InverseDynamics
from DiT.ODE import ODE
from DiT.planner import Planner
from matplotlib import pyplot as plt



#%% Hyperparameters

modality = "S" # whether diffusion predicts only states "S", states and actions "SA", or only actions "A"
env_name = "Quadcopter" # name of the environment in ["Hopper", "Walker", "HalfCheetah", "Quadcopter", "GO1", "GO2","Cartpole","Acrobot"]
proj_name = None # name of the projector in [None, "Ref", "Adm", "SA", "A"]
conditioning = "s0" # attributes on which the diffusion model is conditioned in [None, "s0", "cmd", "s0_cmd"]

extra_name = "" # string to add to the diffusion model to differentiate it from others 
N_samples = 12 # number of sample trajectories to generate
time_limit = None # stops the training after this many seconds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(0) 


#%%

env, model_size, H, N_trajs = make_env(env_name, modality)
proj = load_proj(proj_name, env, device, modality)
print("Device:", device)


attr_d = 17
#%% Load a pretrained DiT
ode = ODE(env, modality=modality,attr_dim=attr_d, device=device, **model_size, projector=proj)
print(ode.projector_name)
assert ode.load(extra=extra_name), f"Model {ode.filename+extra_name} cannot be loaded"
planner = Planner(env, ode)

#%% Load inverse dynamics model

if "S" in modality: # no inverse dynamics for 'A' models since they directly generate admissible trajectories
    ID = InverseDynamics(env)

#%% Conditioning
s0 = env.reset()
if conditioning is None:
    attr = None
elif conditioning == "s0":
    attr = s0.copy()
elif "cmd" in conditioning:
    # cmd = env.sample_command()
    cmd = np.array([1., 0., 0.])
    if conditioning == "s0_cmd":
        attr = np.concatenate((s0, cmd))
    else:
        attr = cmd

#%% Evaluation

out = planner.best_traj(s0, traj_len=H, attr=attr, projector=ode.projector, n_samples_per_s0=N_samples)
if modality == "S":
    sampled_traj = out[0]
    # plt.plot(sampled_traj[:,0],sampled_traj[:,1],'k--')
    # plt.show()
    ID_traj, ID_actions, reward, survival = ID.closest_admissible_traj(sampled_traj)
    env.traj_comparison(sampled_traj, "sampled", ID_traj, "ID", title=ode.filename)

elif modality == "SA":
    sampled_traj, actions = out
    reward, survival, open_loop_traj = open_loop(env, s0, actions[0], attr=attr)
    print(f"{env_name} gets reward of {reward:.2f} and survives {survival*100:.0f}%")
    env.traj_comparison(sampled_traj[0], "sampled", open_loop_traj, "open-loop", title=ode.filename)

elif modality == "A":
    sampled_traj, actions, reward, survival = out
    print(f"{env_name} gets reward of {reward[0]:.2f} and survives {survival[0]*100:.0f}%")
    T = int(survival[0]*(H-1))
    env.plot_traj(sampled_traj[0, :T+1], title=ode.filename)


def simulate_and_linearize_torch(env, X_init, T, eps=1e-4, device="cpu"):
    """
    Simulate env forward with zero actions and compute finite-difference Jacobians (Torch version).

    Args:
        env : simulator with .state and .step(u)
        X_init : (N, n) initial states
        T : number of timesteps
        eps : finite difference epsilon
        device : torch device string

    Returns:
        X : (N, T, n)
        U : (N, T, m)
        Fx : (N, T, n, n)
        Fu : (N, T, n, m)
    """
    #X_init = pred[:, 0, :]
    N, n = X_init.shape
    m = env.action_size
    X = torch.zeros((N, T, n), device=device)
    U = torch.zeros((N, T, m), device=device)
    Fx = torch.zeros((N, T, n, n), device=device)
    Fu = torch.zeros((N, T, n, m), device=device)

    for b in range(N):
        x = X_init[b].detach().cpu().numpy() #X_init[b].detach().numpy() TODO
        for t in range(T):
            u = (torch.zeros((m,), device=device).cpu().numpy())

            env.state = x #.clone().detach().cpu().numpy()
            x_next, *_ = env.step(u)
            #X[b, t] = torch.tensor(x, device=device, dtype=torch.float32)
            X[b, t] = torch.from_numpy(x).float().to(device)

            Fx_b, Fu_b = finite_diff_jacobian_torch(env, x, u, eps, device)
            Fx[b, t] = Fx_b
            Fu[b, t] = Fu_b

            x = x_next.copy()

    return X, U, Fx, Fu


def finite_diff_jacobian_torch(env, x, u, eps=1e-4, device="cpu"):
    """Finite difference Jacobian using Torch tensors."""
    x = torch.tensor(x, dtype=torch.float32, device=device)
    u = torch.tensor(u, dtype=torch.float32, device=device)
    n, m = x.numel(), u.numel()

    f_x = torch.zeros((n, n), device=device)
    f_u = torch.zeros((n, m), device=device)

    env.state = x.cpu().numpy()
    x_nom, *_ = env.step(u.cpu().numpy())
    # x_nom = torch.tensor(x_nom, dtype=torch.float32, device=device)

    # df/dx
    for i in range(n):
        dx = torch.zeros_like(x)
        dx[i] = eps

        env.state = (x + dx).cpu().numpy()
        x_p, *_ = env.step(u.cpu().numpy())
        env.state = (x - dx).cpu().numpy()
        x_m, *_ = env.step(u.cpu().numpy())

        x_p = torch.tensor(x_p, dtype=torch.float32, device=device)
        x_m = torch.tensor(x_m, dtype=torch.float32, device=device)
        f_x[:, i] = (x_p - x_m) / (2 * eps)

    # df/du
    for j in range(m):
        du = torch.zeros_like(u)
        du[j] = eps

        env.state = x.cpu().numpy()
        x_p, *_ = env.step((u + du).cpu().numpy())
        env.state = x.cpu().numpy()
        x_m, *_ = env.step((u - du).cpu().numpy())

        x_p = torch.tensor(x_p, dtype=torch.float32, device=device)
        x_m = torch.tensor(x_m, dtype=torch.float32, device=device)
        f_u[:, j] = (x_p - x_m) / (2 * eps)

    env.state = x.cpu().numpy()
    return f_x, f_u


def iLQR_batch_torch(env, Trajs: torch.Tensor, Ref_Trajs: torch.Tensor = None, max_iters=50, eps=1e-4, alpha=0.5, device="cpu"):
    """
    Torch implementation of batched iLQR using finite-difference Jacobians.
    """

    N, T, n = Trajs.shape
    m = env.action_size
    ref = Ref_Trajs

    Q = torch.eye(n, device=device)
    R = 0.1 * torch.eye(m, device=device)
    Qf = 10 * torch.eye(n, device=device)

    # initialize controls and simulate
    X_init = Trajs[:, 0, :].clone()
    X, U, Fx, Fu = simulate_and_linearize_torch(env, X_init, T, eps, device)

    for it in range(max_iters):
        K = torch.zeros((N, T, m, n), device=device)
        k = torch.zeros((N, T, m), device=device)

        for b in range(N):
            Vx = Qf @ (X[b, -1] - Trajs[b, -1])
            Vxx = Qf.clone()

            for t in reversed(range(T - 1)):
                fx = Fx[b, t]
                fu = Fu[b, t]
                x_err = X[b, t] - Trajs[b, t]
                u = U[b, t]

                Qx = Q @ x_err + fx.T @ Vx
                Qu = R @ u + fu.T @ Vx
                Qxx = Q + fx.T @ Vxx @ fx
                Quu = R + fu.T @ Vxx @ fu
                Qux = fu.T @ Vxx @ fx

                Quu_reg = Quu + 1e-6 * torch.eye(m, device=device)
                k[b, t] = -torch.linalg.solve(Quu_reg, Qu)
                K[b, t] = -torch.linalg.solve(Quu_reg, Qux)

                Vx = Qx + K[b, t].T @ Quu @ k[b, t] + K[b, t].T @ Qu + Qux.T @ k[b, t]
                Vxx = Qxx + K[b, t].T @ Quu @ K[b, t] + K[b, t].T @ Qux + Qux.T @ K[b, t]
                Vxx = 0.5 * (Vxx + Vxx.T)

        # Forward rollout
        X_new = torch.zeros_like(X)
        U_new = torch.zeros_like(U)
        for b in range(N):
            x = X[b, 0].clone()
            for t in range(T - 1):
                dx = x - X[b, t]
                du = alpha * k[b, t] + K[b, t] @ dx
                u = U[b, t] + du
                env.state = x.cpu().numpy()
                x_next, *_ = env.step(u.detach().cpu().numpy())
                X_new[b, t] = x
                U_new[b, t] = u
                x = torch.tensor(x_next, dtype=torch.float32, device=device)
            X_new[b, -1] = x

        # cost check
        cost_prev = trajectory_cost_torch(X, U, Trajs, Q, R, Qf)
        cost_new = trajectory_cost_torch(X_new, U_new, Trajs, Q, R, Qf)
        if torch.mean(torch.abs(cost_new - cost_prev)) < 1e-6:
            break

        X, U = X_new, U_new
        _, _, Fx, Fu = simulate_and_linearize_torch(env, X[:, 0, :], T, eps, device)

    return X, U, K, k


def trajectory_cost_torch(X, U, Xd, Q, R, Qf):
    N, T, n = X.shape
    cost = torch.zeros(N, device=X.device)
    for b in range(N):
        dx = X[b, :-1] - Xd[b, :-1]
        cost[b] += torch.sum((dx @ Q) * dx)
        cost[b] += torch.sum((U[b, :-1] @ R) * U[b, :-1])
        dxT = X[b, -1] - Xd[b, -1]
        cost[b] += (dxT @ Qf @ dxT)
    return cost

#X_feas, U_, K_, k_ = iLQR_batch_torch(env, Trajs=torch.from_numpy(sampled_traj).unsqueeze(0).float() , Ref_Trajs = None, max_iters=50, eps=1e-4, alpha=0.5)
#print(sampled_traj[-1,0:2])
np.savez('quadprojunprojtrajectory.npz', sampled_traj)
# env.traj_comparison(sampled_traj, "sampled", ID_traj, "ID", title=ode.filename, traj_3=X_feas[0,:,:], label_3="feasible")
# np.savez('feasible_traj.npz', X_feas[0,:,:])
