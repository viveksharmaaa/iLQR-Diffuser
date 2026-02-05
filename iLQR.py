
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/sharma/Projects/DDAT/code'])

# “Naive state-diffusion produces kinematically incoherent trajectories that cannot be projected by single-shooting iLQR.”
# “We resolve this by [pick one: spline prefilter / multiple-shooting / action-space diffusion], which dramatically stabilizes the downstream optimal control refinement.”

import torch
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg',force=True)
from utils.loaders import make_env, load_proj

# data_unfeas = np.load("/home/sharma/Projects/DDAT/code/quadprojreftrajectory.npz")
# #data_feas = np.load("/home/sharma/Projects/DDAT/code/feasible_traj.npz")
# # plt.plot(data['arr_0'][:,0],data['arr_0'][:,0],'k--')
# # plt.scatter(data_unfeas['arr_0'][:,0],data_unfeas['arr_0'][:,0])
# #plt.scatter(data_feas['arr_0'][:,0],data_feas['arr_0'][:,0],'k--')
# plt.plot(data_unfeas['arr_0'][:,0],label="x")
# plt.plot(data_unfeas['arr_0'][:,1],label="y")
# plt.plot(data_unfeas['arr_0'][:,2],label="z")
# plt.plot(data_unfeas['arr_0'][:,3],label="v_x")
# plt.plot(data_unfeas['arr_0'][:,4],label="v_y")
# plt.plot(data_unfeas['arr_0'][:,5],label="v_z")

# plt.plot(data_feas['arr_0'][:,0],label="xt")
# plt.plot(data_feas['arr_0'][:,1],label="yt")
# plt.plot(data_feas['arr_0'][:,2],label="zt")

# plt.legend()
# plt.show()

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
    x = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    u = torch.from_numpy(u).to(device=device, dtype=torch.float32)
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
    lam = 1e-6

    Q = torch.diag(torch.tensor([100000, 100000, *([1]*15)])).to(device).float() #torch.eye(n, device=device)
    R = 1000000* torch.eye(m, device=device)
    Qf = 1 * torch.diag(torch.tensor([100000, 100000, *([1]*15)])).to(device).float()  #torch.eye(n, device=device)

    alphas = [1.0, 0.5, 0.25, 0.1, 0.05]  # candidate step sizes
    tol_cost = 1e-6
    #tol_grad = 1e-5
    lam = 1e-6
    diverged = False
    # Initial rollout
    X_init = Trajs[:, 0, :].clone()
    X, U, Fx, Fu = simulate_and_linearize_torch(env, X_init, T, eps, device)
    if not torch.isfinite(Fx).all() or not torch.isfinite(Fu).all():
        print("NaN Jacobians detected, skipping iteration")
        diverged = True
    cost_prev = trajectory_cost_torch(X, U, Trajs, Q, R, Qf)

    for it in range(max_iters):
        diverged = False
        K = torch.zeros((N, T, m, n), device=device)
        k = torch.zeros((N, T, m), device=device)
        print(f"\n=== iLQR Iteration {it} (λ={lam:.1e}) ===")

        # Backward pass
        for b in range(N):
            Vx = Qf @ (X[b, -1] - Trajs[b, -1])
            Vxx = Qf.clone()
            for t in reversed(range(T - 1)):
                fx, fu = Fx[b, t], Fu[b, t]
                x_err, u = X[b, t] - Trajs[b, t], U[b, t]

                Qx = Q @ x_err + fx.T @ Vx
                Qu = R @ u + fu.T @ Vx
                Qxx = Q + fx.T @ Vxx @ fx
                Quu = R + fu.T @ Vxx @ fu
                Qux = fu.T @ Vxx @ fx

                if not torch.isfinite(Fu).all():
                    bad_idx = torch.where(~torch.isfinite(Fu))
                    print(f"[NaN Jacobian] Non-finite Fu at batch {b}, time {t}, indices {bad_idx}")
                    diverged = True
                    break

                # --- Debug: condition numbers and PD check ---
                cond_Quu = torch.linalg.cond(Quu)
                cond_Fu = torch.linalg.cond(fu) if fu.numel() > 0 else torch.tensor(0.)
                if torch.isnan(Quu).any():
                    print(f"[b={b}] NaN detected in Quu at t={t}")
                    diverged = True;
                    break
                if cond_Quu > 1e8:
                    print(f"[b={b}] Ill-conditioned Quu at t={t}: cond={cond_Quu:.2e}")
                if cond_Fu > 1e6:
                    print(f"[b={b}] Ill-conditioned Fu at t={t}: cond={cond_Fu:.2e}")

                # Regularization and invertibility check
                Quu_reg = Quu + lam * torch.eye(m, device=device)
                if torch.isnan(Quu_reg).any() or torch.linalg.cond(Quu_reg) > 1e8:
                    diverged = True
                    break

                try:
                    Quu_inv = torch.linalg.inv(Quu_reg)
                except RuntimeError:
                    diverged = True
                    break

                # Gains
                k[b, t] = -Quu_inv @ Qu
                K[b, t] = -Quu_inv @ Qux

                # Value function update
                Vx = Qx + K[b, t].T @ Quu @ k[b, t] + K[b, t].T @ Qu + Qux.T @ k[b, t]
                Vxx = Qxx + K[b, t].T @ Quu @ K[b, t] + K[b, t].T @ Qux + Qux.T @ K[b, t]
                Vxx = 0.5 * (Vxx + Vxx.T)

                # --- Optional: eigenvalue sanity ---
                eigs = torch.linalg.eigvalsh(Vxx)
                if (eigs < -1e-6).any():
                    print(f"[b={b}] Vxx not PSD at t={t}, min eig={eigs.min():.2e}")

            if diverged:
                break

        if diverged:
            lam *= 10
            print(f"[iter {it}] Diverged in backward pass → increasing λ to {lam:.1e}")
            continue
        print(f"[iter {it}] ✅ Backward pass completed successfully (λ={lam:.1e})")
        # ----- Forward line search -----
        success = False
        best_cost = torch.inf
        best_X, best_U = X.clone(), U.clone()

        for alpha in alphas:
            X_new = torch.zeros_like(X)
            U_new = torch.zeros_like(U)
            for b in range(N):
                x = X[b, 0].clone()
                for t in range(T - 1):
                    dx = x - X[b, t]
                    du = alpha * k[b, t] + K[b, t] @ dx
                    u_new = U[b, t] + du
                    env.state = x.detach().cpu().numpy()
                    x_next, *_ = env.step(u_new.detach().cpu().numpy())
                    X_new[b, t] = x
                    U_new[b, t] = u_new
                    x = torch.tensor(x_next, dtype=torch.float32, device=device)
                X_new[b, -1] = x

            cost_new = trajectory_cost_torch(X_new, U_new, Trajs, Q, R, Qf)
            dcost = (cost_new - cost_prev).mean().item()

            if not torch.isfinite(cost_new).all():
                print(f"[iter {it}] α={alpha:.2f}: NaN cost, rejecting.")
                continue

            if dcost < -tol_cost:
                print(f"[iter {it}] α={alpha:.2f}: cost improved by {dcost:.3e}")
                success = True
                best_cost = cost_new
                best_X, best_U = X_new, U_new
                break
            else:
                print(f"[iter {it}] α={alpha:.2f}: no improvement (ΔJ={dcost:.3e})")

        # ----- Accept / reject step -----
        if success:
            X, U = best_X, best_U
            cost_prev = best_cost
            lam = max(lam / 2, 1e-6)
        else:
            lam *= 10
            print(f"[iter {it}] Line search failed → increasing λ to {lam:.1e}")

        # Convergence check
        if lam > 1e6:
            print(f"[iter {it}] λ too large → terminating (unstable).")
            break
        if torch.mean(torch.abs(cost_new - cost_prev)) < tol_cost:
            print(f"[iter {it}] Converged (ΔJ < {tol_cost}).")
            break

        X, U = X_new, U_new
        _, _, Fx, Fu = simulate_and_linearize_torch(env, X[:, 0, :], T, eps, device)
        plt.plot(X[0, :,0], X[0,:, 1],label="feasible")
        plt.scatter(X[0, :, 0].detach().cpu(), X[0, :, 1].detach().cpu())
        plt.plot(Trajs[0, :,0], Trajs[0,:, 1],label="infeasible")
        plt.scatter(Trajs[0, :, 0].detach().cpu(), Trajs[0, :, 1].detach().cpu())
        plt.show()

    #plt.plot(X[:, 0], X[:, 1])

    return X, U, K, k


def ilqr_multiple_shooting(
    env,
    Trajs,          # (N,T,n) reference (infeasible) traj we want to project toward
    eps,
    device,
    max_iters=50,
    M=20,            # <-- number of shooting segments
    alphas=[1.0, 0.5, 0.25, 0.1, 0.05]
):

    N, T, n = Trajs.shape
    m = env.action_size
    lam = 1e-6
    tol_cost = 1e-6

    assert T % M == 0, "for now assume horizon divisible by #segments"
    Hs = T // M  # segment length

    # Cost weights
    Q   = torch.eye(n, device=device)
    R   = 0.1 * torch.eye(m, device=device)
    Qf  = 10.0 * torch.eye(n, device=device)
    Qdef = 100.0 * torch.eye(n, device=device)  ### NEW: defect penalty at segment boundaries

    # Q = torch.diag(torch.tensor([100000, 100000, *([1]*15)])).to(device).float() #torch.eye(n, device=device)
    # R = 1000000* torch.eye(m, device=device)
    # Qf = 1 * torch.diag(torch.tensor([100000, 100000, *([1]*15)])).to(device).float()  #torch.eye(n, device=device)

    # ---------------------------
    # Helper: rollout + linearize per segment
    # ---------------------------
    def rollout_segments(X_segstarts, U_full):
        """
        X_segstarts:  (N,M,n)   decision vars: start state of each shooting segment
        U_full:       (N,T,m)   nominal control sequence (full horizon)

        returns:
            X_full: (N,T,n) concatenated simulated states
            Fx_full, Fu_full: (N,T-1,n,n)/(N,T-1,n,m) local linear models along X_full,U_full
            seg_end_states: (N,M,n) the simulated final state of each segment
        """
        X_full = torch.zeros((N, T, n), device=device)
        Fx_full = torch.zeros((N, T-1, n, n), device=device)
        Fu_full = torch.zeros((N, T-1, n, m), device=device)
        seg_end_states = torch.zeros((N, M, n), device=device)

        for b in range(N):
            idx_t = 0
            for s in range(M):
                x = X_segstarts[b, s].clone()  # start of this segment
                for k in range(Hs):
                    t_idx = idx_t + k
                    if k < Hs - 1:
                        u = U_full[b, t_idx]
                        # we need local linearization around (x,u)
                        env.state = x.detach().cpu().numpy()
                        x_next, *_ = env.step(u.detach().cpu().numpy())
                        x_next = torch.tensor(x_next, dtype=torch.float32, device=device)

                        # finite-diff Jacobians around (x,u)
                        fx, fu =  finite_diff_jacobian_torch(env, x.detach().cpu().numpy() , u.detach().cpu().numpy(), eps, device)  # you'll define this like simulate_and_linearize_torch did internally
                        Fx_full[b, t_idx] = fx
                        Fu_full[b, t_idx] = fu

                        X_full[b, t_idx] = x
                        x = x_next
                    else:
                        # last state of the segment goes in X_full as well
                        X_full[b, t_idx] = x
                        # advance one more step to get true end-of-segment state
                        u_last = U_full[b, t_idx]
                        env.state = x.detach().cpu().numpy()
                        x_next, *_ = env.step(u_last.detach().cpu().numpy())
                        x_next = torch.tensor(x_next, dtype=torch.float32, device=device)
                        seg_end_states[b, s] = x_next
                        # NOTE: we do NOT store Jacobians at this "phantom" step
                idx_t += Hs

        # also assign final terminal state in X_full
        X_full[:, -1, :] = seg_end_states[:, -1, :]

        return X_full, Fx_full, Fu_full, seg_end_states

    # ---------------------------
    # Helper: cost with multiple-shooting defects
    # ---------------------------
    def ms_cost(X_full, U_full, Trajs_ref, X_segstarts, seg_end_states):
        """
        Augmented cost:
          tracking+control cost along horizon +
          terminal cost +
          defect penalties between segments
        """
        N, T, n = X_full.shape
        cost = torch.zeros(N, device=device)

        # running cost
        for t in range(T-1):
            x_err = X_full[:, t] - Trajs_ref[:, t]
            u     = U_full[:, t]
            for b in range(N):
                cost[b] += x_err[b] @ Q @ x_err[b] + u[b] @ R @ u[b]

        # terminal cost
        xT_err = X_full[:, -1] - Trajs_ref[:, -1]
        for b in range(N):
            cost[b] += xT_err[b] @ Qf @ xT_err[b]

        # defect penalties at segment boundaries
        # for s = 0...(M-2), enforce next segment's declared start == dyn end of current segment
        for s in range(M-1):
            d = X_segstarts[:, s+1] - seg_end_states[:, s]   # (N,n)
            for b in range(N):
                cost[b] += d[b] @ Qdef @ d[b]

        return cost.mean()

    # ------------------------------------------------
    # Initialize decision vars
    # ------------------------------------------------
    # We'll initialize each segment start from the provided (possibly infeasible) Trajs
    # and initialize U as zeros.
    X_segstarts = torch.zeros((N, M, n), device=device)
    for s in range(M):
        X_segstarts[:, s, :] = Trajs[:, s*Hs, :]

    U = torch.zeros((N, T, m), dtype=torch.float32, device=device)

    # Initial rollout
    # (We'll steal your finite diff logic, but above I assumed a helper finite_diff_jacobians; adapt that to your codebase.)
    X, Fx, Fu, seg_end_states = rollout_segments(X_segstarts, U)
    cost_prev = ms_cost(X, U, Trajs, X_segstarts, seg_end_states)

    for it in range(max_iters):
        diverged = False

        # We'll build feedback gains for *every timestep* like before.
        K_u = torch.zeros((N, T, m, n), device=device)
        k_u = torch.zeros((N, T, m), device=device)

        # We'll ALSO build gains for the shooting node states, because those are now decision vars.
        # Each shooting node s has its own "perturbation direction".
        K_xseg = torch.zeros((N, M, n, n), device=device)   # linear feedback on segment start state
        k_xseg = torch.zeros((N, M, n), device=device)      # feedforward shift for segment start

        # -------- Backward pass logic --------
        # We go backward in time like standard iLQR, but:
        # - Inside each segment, same recursions.
        # - At segment boundary s→s+1, we add defect penalty to value function at the *start* of segment s+1.

        # We will accumulate Vx,Vxx at each timestep going backward.
        Vx  = torch.zeros((N, n), device=device)
        Vxx = torch.zeros((N, n, n), device=device)

        # Start from terminal:
        xT_err = X[:, -1] - Trajs[:, -1]
        Vx = (Qf @ xT_err.transpose(1,0)).transpose(1,0)        # (N,n) = Qf (xT - xT_ref)
        Vxx = Qf.expand(N, n, n).clone()                        # (N,n,n)

        # We'll walk t = T-2 down to 0. We also need to know which segment we're in.
        for t in reversed(range(T-1)):
            # which segment does this timestep belong to?
            s = t // Hs

            fx = Fx[:, t]    # (N,n,n)
            fu = Fu[:, t]    # (N,n,m)
            x_err = X[:, t] - Trajs[:, t]
            u     = U[:, t]

            # Build Qx, Qu, Qxx, Quu, Qux *for each batch b*
            # We'll solve them batch-wise similarly to your loop.
            for b in range(N):
                fx_b = fx[b]
                fu_b = fu[b]
                Vx_b = Vx[b]
                Vxx_b= Vxx[b]

                Qx_b  = Q @ x_err[b] + fx_b.T @ Vx_b
                Qu_b  = R @ u[b]     + fu_b.T @ Vx_b
                Qxx_b = Q + fx_b.T @ Vxx_b @ fx_b
                Quu_b = R + fu_b.T @ Vxx_b @ fu_b
                Qux_b = fu_b.T @ Vxx_b @ fx_b

                # Regularize for stability
                Quu_reg_b = Quu_b + lam * torch.eye(m, device=device)

                # Try inverse
                try:
                    Quu_inv_b = torch.linalg.inv(Quu_reg_b)
                except RuntimeError:
                    diverged = True
                    break

                # Standard gains for control
                k_u[b, t] = -Quu_inv_b @ Qu_b
                K_u[b, t] = -Quu_inv_b @ Qux_b

                # Update Value function (classic iLQR algebra)
                Vx_update = (
                    Qx_b
                    + K_u[b, t].T @ Quu_b @ k_u[b, t]
                    + K_u[b, t].T @ Qu_b
                    + Qux_b.T @ k_u[b, t]
                )
                Vxx_update = (
                    Qxx_b
                    + K_u[b, t].T @ Quu_b @ K_u[b, t]
                    + K_u[b, t].T @ Qux_b
                    + Qux_b.T @ K_u[b, t]
                )
                Vxx_update = 0.5 * (Vxx_update + Vxx_update.T)

                Vx[b]  = Vx_update
                Vxx[b] = Vxx_update

            if diverged:
                break

            # --- If t is the FIRST timestep of a segment (t % Hs == 0),
            # we also need to "push back" value function info to the free shooting node X_segstarts[:,s].
            if t % Hs == 0:
                # incorporate defect penalty that links seg_end_states[s-1] -> X_segstarts[s]
                # Careful with s=0 (no previous segment).
                if s > 0:
                    # defect d_{s-1} = X_segstarts[s] - seg_end_states[s-1]
                    # cost += d^T Qdef d
                    # gradient wrt X_segstarts[s] is 2 Qdef d
                    # Hessian wrt X_segstarts[s] is 2 Qdef
                    d = X_segstarts[:, s] - seg_end_states[:, s-1]   # (N,n)

                    for b in range(N):
                        Vx[b]  += 2.0 * (Qdef @ d[b])
                        Vxx[b] += 2.0 * Qdef  # symmetric

                # store "gains" for this shooting node like we do for controls
                # this is analogous to k_u/K_u, but w.r.t changing X_segstarts[s]
                # Heuristically: we’ll do a Gauss-Newton style step on node states.
                for b in range(N):
                    # simple gradient step for node correction
                    # (no coupling across batches here)
                    try:
                        Vxx_reg = Vxx[b] + lam * torch.eye(n, device=device)
                        dx_node = -torch.linalg.solve(Vxx_reg, Vx[b])
                    except RuntimeError:
                        diverged = True
                        break
                    k_xseg[b, s] = dx_node
                    K_xseg[b, s] = torch.zeros((n, n), device=device)  # (we're not doing feedback coupling across nodes here)
                if diverged:
                    break

        if diverged:
            lam *= 10
            print(f"[iter {it}] Diverged in backward pass → λ={lam:.1e}")
            continue
        print(f"[iter {it}] ✅ Backward pass OK (λ={lam:.1e})")

        # -------- Forward line search (multiple shooting style) --------
        success = False
        best_cost = torch.inf
        best_X     = X.clone()
        best_U     = U.clone()
        best_nodes = X_segstarts.clone()
        best_segends = seg_end_states.clone()

        for alpha in alphas:
            # propose NEW shooting nodes (segment start states)
            X_segstarts_new = torch.zeros_like(X_segstarts)
            for s in range(M):
                # x_s_new = x_s_old + alpha * k_xseg[s]
                X_segstarts_new[:, s] = X_segstarts[:, s] + alpha * k_xseg[:, s]

            # now closed-loop inside each segment using k_u,K_u like standard iLQR
            U_new = torch.zeros_like(U)
            X_new = torch.zeros_like(X)

            # We'll regenerate by rolling forward segment by segment with feedback in the loop
            seg_end_states_new = torch.zeros_like(seg_end_states)

            for b in range(N):
                idx_t = 0
                for s in range(M):
                    x = X_segstarts_new[b, s].clone()
                    for k_step in range(Hs):
                        t_idx = idx_t + k_step

                        if t_idx < T-1:
                            dx = x - X[b, t_idx]
                            du = alpha * k_u[b, t_idx] + K_u[b, t_idx] @ dx
                            u_prop = U[b, t_idx] + du
                        else:
                            u_prop = U[b, t_idx]

                        # step env
                        env.state = x.detach().cpu().numpy()
                        x_next, *_ = env.step(u_prop.detach().cpu().numpy())
                        x_next = torch.tensor(x_next, dtype=torch.float32, device=device)

                        X_new[b, t_idx] = x
                        U_new[b, t_idx] = u_prop
                        x = x_next

                    # finished segment -> record end state
                    seg_end_states_new[b, s] = x.clone()
                    idx_t += Hs

                X_new[b, -1] = seg_end_states_new[b, -1].clone()

            cost_new = ms_cost(X_new, U_new, Trajs, X_segstarts_new, seg_end_states_new)
            dcost = (cost_new - cost_prev).item()

            if not torch.isfinite(cost_new):
                print(f"[iter {it}] α={alpha:.2f}: NaN cost, reject.")
                continue

            if dcost < -tol_cost:
                print(f"[iter {it}] α={alpha:.2f}: cost improved by {dcost:.3e}")
                success = True
                best_cost      = cost_new
                best_X         = X_new
                best_U         = U_new
                best_nodes     = X_segstarts_new
                best_segends   = seg_end_states_new
                break
            else:
                print(f"[iter {it}] α={alpha:.2f}: no improvement (ΔJ={dcost:.3e})")

        # ----- Accept / reject -----
        if success:
            X              = best_X
            U              = best_U
            X_segstarts    = best_nodes
            seg_end_states = best_segends
            cost_prev      = best_cost
            lam            = max(lam / 2, 1e-6)
        else:
            lam *= 10
            print(f"[iter {it}] Line search failed → λ={lam:.1e}")

        # stopping conditions
        if lam > 1e6:
            print(f"[iter {it}] λ too large → terminating.")
            break

        if (ms_cost(X, U, Trajs, X_segstarts, seg_end_states) - cost_prev).abs() < tol_cost:
            print(f"[iter {it}] Converged (ΔJ < {tol_cost}).")
            break

        # debug plotting for batch 0
        plt.plot(X[0, :, 0].detach().cpu(), X[0, :, 1].detach().cpu(), label="feasible(ms)")
        plt.plot(Trajs[0, :, 0].detach().cpu(), Trajs[0, :, 1].detach().cpu(), label="ref")
        plt.legend()
        plt.show()

    return X, U, X_segstarts



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

if __name__ == '__main__':
    data_unfeas = np.load("/home/sharma/Projects/DDAT/code/quadprojreftrajectory.npz") #
    # print(data_unfeas["arr_0"].shape)
    plt.figure(1)
    device = "cuda:0"
    plt.plot(data_unfeas["arr_0"][:, 0], data_unfeas["arr_0"][:, 1])
    plt.scatter(data_unfeas["arr_0"][:, 0], data_unfeas["arr_0"][:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title('Dataset')
    # plt.plot(data_unfeas['arr_0'][:,0],label="x")
    # plt.plot(data_unfeas['arr_0'][:,1],label="y")
    # plt.plot(data_unfeas['arr_0'][:,2],label="z")
    # plt.plot(data_unfeas['arr_0'][:,7],label="v_x")
    # plt.plot(data_unfeas['arr_0'][:,8],label="v_y")
    # plt.plot(data_unfeas['arr_0'][:,9],label="v_z")
    # plt.legend()
    # plt.show()
    Trajs = torch.from_numpy(data_unfeas["arr_0"]).unsqueeze(0).float().to(device)
    env, model_size, H, N_trajs = make_env("Quadcopter", "S")
    #X_feas, U_feas, K_feas, k_feas = iLQR_batch_torch(env,Trajs, Ref_Trajs= None, max_iters = 50, eps = 1e-4, alpha = 0.5)
    Xm_feas, Um, Xm_segstarts = ilqr_multiple_shooting(env,Trajs,eps=1e-4,device=device,max_iters=50,M=50)
    #print(Xm_feas)

    plt.figure(2)
    #plt.plot(X_feas[:, 0], X_feas[:, 1])
    plt.plot(Xm_feas[0,:, 0].detach().cpu().numpy(), Xm_feas[0,:, 1].detach().cpu().numpy(),'--')
    plt.plot(data_unfeas["arr_0"][:, 0], data_unfeas["arr_0"][:, 1])

    plt.scatter(Xm_feas[0,:, 0].detach().cpu().numpy(), Xm_feas[0,:, 1].detach().cpu().numpy())
    plt.scatter(data_unfeas["arr_0"][:, 0], data_unfeas["arr_0"][:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

