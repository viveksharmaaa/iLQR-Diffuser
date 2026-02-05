import os, itertools, numpy as np
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "palatino"
plt.rcParams.update({
    "font.size": 12,         # default text size
    "axes.titlesize": 16,    # title
    "axes.labelsize": 14,    # x and y labels
    "xtick.labelsize": 12,   # x tick labels
    "ytick.labelsize": 12,   # y tick labels
    "legend.fontsize": 12,   # legend
    "font.weight": "bold",
})


# ----------------------- Parameters -----------------------
def quad_params():
    m = 1.0
    g = -9.81                   # matches the sign convention in your figure
    Ix, Iy, Iz = 0.5, 0.1, 0.3  # kg·m^2
    return m, g, Ix, Iy, Iz

# ----------------------- Dynamics (continuous time) -----------------------
def f_continuous(x, u):
    m, g, Ix, Iy, Iz = quad_params()
    px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r = x
    u1, u2, u3, u4 = u

    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    # Kinematics
    dx, dy, dz = vx, vy, vz

    # Euler angle rates (ZYX)
    eps = 1e-9
    phi_dot   = q * sphi / (cth + eps) + r * cphi / (cth + eps)
    theta_dot = q * cphi - r * sphi
    psi_dot   = p + q * sphi * (sth/(cth + eps)) + r * cphi * (sth/(cth + eps))

    # Translational accelerations (matches your figure)
    ax = (u1/m) * (sphi * spsi + cphi * cpsi * sth)
    ay = (u1/m) * (-sphi * cpsi + cphi * spsi * sth)
    az = g + (u1/m) * (cphi * cth)

    # Body rate dynamics
    p_dot = ((Iy - Iz) / Ix) * q * r + u2 / Ix
    q_dot = ((Iz - Ix) / Iy) * p * r + u3 / Iy
    r_dot = ((Ix - Iy) / Iz) * p * q + u4 / Iz

    return np.array([dx, dy, dz, phi_dot, theta_dot, psi_dot,
                     ax, ay, az, p_dot, q_dot, r_dot])

# ----------------------- RK4 discretization -----------------------
def rk4_step(x, u, dt):
    k1 = f_continuous(x, u)
    k2 = f_continuous(x + 0.5*dt*k1, u)
    k3 = f_continuous(x + 0.5*dt*k2, u)
    k4 = f_continuous(x + dt*k3, u)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ----------------------- Simulation -----------------------
def simulate(x0, U, dt):
    N = U.shape[0]
    X = np.zeros((N+1, x0.size))
    X[0] = x0
    for k in range(N):
        X[k+1] = rk4_step(X[k], U[k], dt)
    return X

# ----------------------- Jacobians (finite-difference) -----------------------
def finite_diff_jacobian(f_disc, x, u, eps=1e-5):
    n, m = x.size, u.size
    fx = np.zeros((n, n))
    fu = np.zeros((n, m))
    f0 = f_disc(x, u)
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        fx[:, i] = (f_disc(x + dx, u) - f0) / eps
    for j in range(m):
        du = np.zeros(m); du[j] = eps
        fu[:, j] = (f_disc(x, u + du) - f0) / eps
    return fx, fu

# ----------------------- iLQR -----------------------
def ilqr(x0, x_ref, u_init, Q, R, Qf, dt, max_iters=60, lam_init=1.0,
         alpha_list=(1.0, 0.5, 0.25, 0.1, 0.03, 0.01),
         u_bounds=None):
    """
    u_bounds: tuple (umin, umax) or None; if provided, both are 4-d arrays applied by clipping.
    """
    def fdisc(x, u): return rk4_step(x, u, dt)

    N, n, m = u_init.shape[0], x0.size, u_init.shape[1]
    U = u_init.copy()
    X = simulate(x0, U, dt)

    def cost(X, U):
        c = 0.0
        for k in range(N):
            dx = X[k] - x_ref[k]
            c += dx @ Q @ dx + U[k] @ R @ U[k]
        dxN = X[N] - x_ref[N]
        c += dxN @ Qf @ dxN
        return c

    J = cost(X, U)
    lam = lam_init

    for _ in range(max_iters):
        # Backward pass
        Vx  = Qf @ (X[N] - x_ref[N])
        Vxx = Qf.copy()

        Ks = np.zeros((N, m, n))
        ks = np.zeros((N, m))

        diverged = False
        for k in reversed(range(N)):
            fx, fu = finite_diff_jacobian(fdisc, X[k], U[k])

            Qx  = Q @ (X[k] - x_ref[k]) + fx.T @ Vx
            Qu  = R @ U[k] + fu.T @ Vx
            Qxx = Q + fx.T @ Vxx @ fx
            Quu = R + fu.T @ Vxx @ fu
            Qux = fu.T @ Vxx @ fx

            # Regularize
            Quu_reg = Quu + lam * np.eye(m)
            try:
                Quu_inv = np.linalg.inv(Quu_reg)
            except np.linalg.LinAlgError:
                diverged = True
                break

            K   = -Quu_inv @ Qux
            kff = -Quu_inv @ Qu

            Ks[k], ks[k] = K, kff

            Vx  = Qx + K.T @ Quu @ kff + K.T @ Qu + Qux.T @ kff
            Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
            Vxx = 0.5 * (Vxx + Vxx.T)  # symmetrize

        if diverged:
            lam *= 10.0
            continue

        # Forward line search
        improved = False
        for alpha in alpha_list:
            Un = np.zeros_like(U)
            Xn = np.zeros_like(X)
            Xn[0] = x0
            for k in range(N):
                du = alpha * ks[k] + Ks[k] @ (Xn[k] - X[k])
                Un[k] = U[k] + du
                if u_bounds is not None:
                    umin, umax = u_bounds
                    Un[k] = np.clip(Un[k], umin, umax)
                Xn[k+1] = rk4_step(Xn[k], Un[k], dt)
            Jn = cost(Xn, Un)
            if Jn < J:
                U, X, J = Un, Xn, Jn
                lam = max(lam / 2.0, 1e-6)
                improved = True
                break

        if not improved:
            lam *= 10.0
        else:
            # convergence test (simple)
            if abs(Jn - J) < 1e-4:
                break

    return X, U, J

# ---------- helpers ----------
def action_bounds():
    # choose sensible bounds for thrust/moments
    # (tune to your platform if you have real limits)
    umin = 10*np.array([0.0,  -2.5, -2.5, -2.5])
    umax = 10*np.array([20.0,  2.5,  2.5,  2.5])
    return umin, umax

def normalize_actions(U, umin, umax):
    # elementwise affine map to [-1, 1], then clip
    U_n = 2.0 * (U - umin) / (umax - umin) - 1.0
    return np.clip(U_n, -1.0, 1.0)

def ref_circle(T, dt, R=2.0, z=1.5):
    t = np.arange(0, T + 1e-9, dt); N = t.size - 1
    w = 2*np.pi / T
    x_ref = np.zeros((N+1, 12))
    x_ref[:,0] = R*np.cos(w*t)
    x_ref[:,1] = R*np.sin(w*t)
    x_ref[:,2] = z
    # hover-thrust baseline
    m, g, *_ = quad_params()
    u_ref = np.tile(np.array([-m*g, 0, 0, 0]), (N,1))
    return t, x_ref, u_ref

def ref_line(T, dt, p0=(0,0,1.5), p1=(2,0,1.5)):
    t = np.arange(0, T + 1e-9, dt); N = t.size - 1
    x_ref = np.zeros((N+1, 12))
    p0 = np.array(p0); p1 = np.array(p1)
    for k, tau in enumerate(np.linspace(0,1,N+1)):
        p = (1-tau)*p0 + tau*p1
        x_ref[k,0:3] = p
    m, g, *_ = quad_params()
    u_ref = np.tile(np.array([-m*g, 0, 0, 0]), (N,1))
    return t, x_ref, u_ref

def ref_hover(T, dt, p=(0,0,1.5)):
    t = np.arange(0, T + 1e-9, dt); N = t.size - 1
    x_ref = np.zeros((N+1, 12))
    x_ref[:,0:3] = np.array(p)
    m, g, *_ = quad_params()
    u_ref = np.tile(np.array([-m*g, 0, 0, 0]), (N,1))
    return t, x_ref, u_ref

# Pack references
REFS = {
    "circle": lambda T,dt: ref_circle(T,dt,R=2.0,z=1.5),
    "line":   lambda T,dt: ref_line(T,dt,p0=(0,0,1.2), p1=(2,1.0,1.5)),
    "hover":  lambda T,dt: ref_hover(T,dt,p=(0.5,-0.5,1.5)),
}

#------------------plot-------------------

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def _equalize_3d_axes(ax, X, Xr):
    xs = np.concatenate([X[:,0], Xr[:,0]])
    ys = np.concatenate([X[:,1], Xr[:,1]])
    zs = np.concatenate([X[:,2], Xr[:,2]])
    xmid, ymid, zmid = xs.mean(), ys.mean(), zs.mean()
    r = max(xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()) / 2.0
    ax.set_xlim(xmid-r, xmid+r)
    ax.set_ylim(ymid-r, ymid+r)
    ax.set_zlim(zmid-r, zmid+r)

def plot_and_save(t, X, U, x_ref, out_png_prefix):
    """
    Saves:
      - {prefix}_pos.png         : x,y,z vs reference
      - {prefix}_inputs.png      : u1..u4 vs time
      - {prefix}_traj3d.png      : 3D (x,y,z)
      - {prefix}_traj_xy.png     : XY projection
    """
    # 1) positions vs ref
    fig1 = plt.figure(figsize=(8,6))
    ax = fig1.add_subplot(311); ax.plot(t, X[:,0], label="x"); ax.plot(t, x_ref[:,0], "--", label="x_ref"); ax.set_ylabel("x [m]"); ax.legend()
    ax = fig1.add_subplot(312); ax.plot(t, X[:,1], label="y"); ax.plot(t, x_ref[:,1], "--", label="y_ref"); ax.set_ylabel("y [m]"); ax.legend()
    ax = fig1.add_subplot(313); ax.plot(t, X[:,2], label="z"); ax.plot(t, x_ref[:,2], "--", label="z_ref"); ax.set_ylabel("z [m]"); ax.set_xlabel("time [s]"); ax.legend()
    fig1.suptitle("Position tracking"); fig1.tight_layout()
    fig1.savefig(out_png_prefix + "_pos.png", dpi=200)
    plt.close(fig1)

    # 2) inputs vs time
    fig2 = plt.figure(figsize=(8,6))
    ax = fig2.add_subplot(411); ax.plot(t[:-1], U[:,0]); ax.set_ylabel("u1 (thrust)")
    ax = fig2.add_subplot(412); ax.plot(t[:-1], U[:,1]); ax.set_ylabel("u2 (Mx)")
    ax = fig2.add_subplot(413); ax.plot(t[:-1], U[:,2]); ax.set_ylabel("u3 (My)")
    ax = fig2.add_subplot(414); ax.plot(t[:-1], U[:,3]); ax.set_ylabel("u4 (Mz)"); ax.set_xlabel("time [s]")
    fig2.suptitle("Control inputs"); fig2.tight_layout()
    fig2.savefig(out_png_prefix + "_inputs.png", dpi=200)
    plt.close(fig2)

    # 3) 3D trajectory
    fig3 = plt.figure(figsize=(7,6))
    ax3d = fig3.add_subplot(111, projection="3d")
    ax3d.plot3D(X[:,0], x_ref[:,1]*0 + X[:,1], X[:,2], label="traj")
    ax3d.plot3D(x_ref[:,0], x_ref[:,1], x_ref[:,2], "--", label="ref")
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]"); ax3d.set_title("3D trajectory")
    _equalize_3d_axes(ax3d, X, x_ref)
    ax3d.legend()
    fig3.tight_layout()
    fig3.savefig(out_png_prefix + "_traj3d.png", dpi=200)
    plt.close(fig3)

    # 4) XY projection
    fig4 = plt.figure(figsize=(6,5))
    axxy = fig4.add_subplot(111)
    axxy.plot(X[:,0], X[:,1], label="traj")
    axxy.plot(x_ref[:,0], x_ref[:,1], "--", label="ref")
    axxy.set_aspect("equal", adjustable="box")
    axxy.set_xlabel("x [m]"); axxy.set_ylabel("y [m]"); axxy.set_title("XY trajectory"); axxy.legend()
    fig4.tight_layout()
    fig4.savefig(out_png_prefix + "_traj_xy.png", dpi=200)
    plt.close(fig4)


# ---------- sweep & save ----------
def sweep_and_save(
    out_dir="runs_ilqr",
    T=8.0,
    dt=0.02,
    initials=None,
    refs=REFS,
):
    os.makedirs(out_dir, exist_ok=True)
    Nsave = 0

    # defaults: a few initial states (x,y,z, phi,th,psi, vx,vy,vz,p,q,r)
    if initials is None:
        initials = [
            np.array([0.0, 0.0, 0.5,   0,0,0,   0,0,0,   0,0,0]),
            np.array([0.3,-0.3, 0.8,   0.05,-0.04,0.0,  0,0,0,  0,0,0]),
            np.array([-0.5,0.4, 1.0,   -0.03,0.02,0.0,  0.1,-0.05,0,  0,0,0]),
        ]

    # cost weights (same as earlier)
    Q  = np.diag([50, 50, 80, 5, 5, 5,  2, 2, 4,  1, 1, 1])
    R  = np.diag([1e-4, 1e-3, 1e-3, 1e-3])
    Qf = np.diag([200, 200, 300, 10, 10, 10, 5, 5, 10, 2, 2, 2])

    umin, umax = action_bounds()
    bounds = (umin, umax)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for ref_name, ref_fn in refs.items():
        t, x_ref, u_ref = ref_fn(T, dt)
        N = t.size - 1

        for i, x0 in enumerate(initials):
            # initial control guess = reference hover thrust
            U0 = u_ref.copy()

            # run iLQR
            X_opt, U_opt, J = ilqr(
                x0=x0,
                x_ref=x_ref,
                u_init=U0,
                Q=Q, R=R, Qf=Qf,
                dt=dt,
                max_iters=60,
                u_bounds=bounds
            )

            # normalize actions to [-1,1] for saving
            #U_norm = normalize_actions(U_opt, umin, umax)

            # save to npz
            fname = f"{ref_name}_ic{i}_{ts}.npz"
            path = os.path.join(out_dir, fname)
            # np.savez_compressed(
            #     path,
            #     t=t,
            #     X=X_opt,
            #     U=U_opt,
            #     U_norm=U_norm,
            #     x_ref=x_ref,
            #     u_ref=u_ref,
            #     umin=umin,
            #     umax=umax,
            #     meta=np.array([f"ref={ref_name}", f"ic_index={i}", f"cost={J}"], dtype=object)
            # )
            print(f"saved: {path}")
            dataset = {"Trajs": X_opt[None, ...], "Actions": normalize_actions(U_opt,umin,umax)[None, ...]}
            np.savez_compressed(path, **dataset)
            # also save plots for quick inspection
            png_prefix = path[:-4]  # drop .npz
            plot_and_save(t, X_opt, U_opt, x_ref, png_prefix)
            print(f"saved plots: {png_prefix}_*.png")
            Nsave += 1
    print(f"done. saved {Nsave} rollouts in '{out_dir}'")

# ---- run the sweep ----
if __name__ == "__main__":
    sweep_and_save(out_dir="runs_ilqr", T=8.0, dt=0.02)
    #dataset = np.load("runs_ilqr/traj.npy")
