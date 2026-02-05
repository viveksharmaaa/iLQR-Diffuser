#!/usr/bin/env python3
# iLQR for a 3D quadrotor (12-state, 4-input) tracking a trajectory.
# State: [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
# Input: [u1 (thrust, N), u2 (Mx), u3 (My), u4 (Mz)]

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

# ----------------------- Reference (circle at constant z) -----------------------
def make_reference(T=8.0, dt=0.02):
    t = np.arange(0.0, T + 1e-9, dt)
    N = t.size - 1
    R = 2.0
    omega = 2*np.pi / T
    px = R * np.cos(omega * t)
    py = R * np.sin(omega * t)
    pz = 1.5 * np.ones_like(t)
    psi = np.zeros_like(t)  # keep yaw ~ 0

    x_ref = np.zeros((N+1, 12))
    x_ref[:, 0] = px
    x_ref[:, 1] = py
    x_ref[:, 2] = pz
    x_ref[:, 5] = psi

    # Hover thrust baseline: u1_ref = -m*g (positive with g<0 here)
    m, g, *_ = quad_params()
    u_ref = np.tile(np.array([-m*g, 0.0, 0.0, 0.0]), (N, 1))
    return t, x_ref, u_ref

# ----------------------- Main -----------------------
if __name__ == "__main__":
    dt = 0.02
    t, x_ref, u_ref = make_reference(T=8.0, dt=dt)
    N = t.size - 1

    # Initial state
    x0 = np.zeros(12)
    x0[2] = 0.5  # start a little below ref altitude

    # Initial control guess
    U0 = u_ref.copy()

    # Costs
    Q  = np.diag([50, 50, 80, 5, 5, 5,  2, 2, 4,  1, 1, 1])
    R  = np.diag([1e-4, 1e-3, 1e-3, 1e-3])
    Qf = np.diag([200, 200, 300, 10, 10, 10, 5, 5, 10, 2, 2, 2])

    # Optional actuator bounds (uncomment if desired)
    # m, g, *_ = quad_params()
    # umin = np.array([0.0, -2.5, -2.5, -2.5])
    # umax = np.array([20.0,  2.5,  2.5,  2.5])
    # bounds = (umin, umax)
    bounds = None

    X_opt, U_opt, J = ilqr(x0, x_ref, U0, Q, R, Qf, dt, max_iters=60, u_bounds=bounds)

    # ----------------------- Plots -----------------------
    plt.figure()
    plt.plot(t, X_opt[:, 0], label="x")
    plt.plot(t, X_opt[:, 1], label="y")
    plt.plot(t, X_opt[:, 2], label="z")
    plt.plot(t, x_ref[:, 0], "--", label=r"$x_{ref}$")
    plt.plot(t, x_ref[:, 1], "--", label=r"$y_{ref}$")
    plt.plot(t, x_ref[:, 2], "--", label=r"$z_{ref}$")
    plt.xlabel("time [s]"); plt.ylabel("position [m]"); plt.legend(); plt.title("Positions vs reference"); plt.tight_layout()

    plt.figure()
    plt.plot(t, X_opt[:, 3], label=r"$\phi$")
    plt.plot(t, X_opt[:, 4], label=r"$\theta$")
    plt.plot(t, X_opt[:, 5], label=r"$\psi$")
    plt.xlabel("time [s]"); plt.ylabel("angles [rad]"); plt.legend(); plt.title("Attitude"); plt.tight_layout()

    plt.figure()
    plt.plot(t[:-1], U_opt[:, 0], label=r"$u_1 (thrust)$")
    plt.plot(t[:-1], U_opt[:, 1], label=r"$u_2 (M_x)$")
    plt.plot(t[:-1], U_opt[:, 2], label=r"$u_3 (M_y)$")
    plt.plot(t[:-1], U_opt[:, 3], label=r"$u_4 (M_z)$")
    plt.xlabel("time [s]"); plt.ylabel("inputs (SI)"); plt.legend(); plt.title("Control inputs"); plt.tight_layout()

    plt.figure()
    plt.plot(X_opt[:, 0], X_opt[:, 1], label="traj")
    plt.plot(x_ref[:, 0], x_ref[:, 1], "--", label="ref")
    plt.axis("equal"); plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.legend(); plt.title("XY trajectory"); plt.tight_layout()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot3D(X_opt[:, 0], X_opt[:, 1], X_opt[:, 2], label="traj")
    ax.plot3D(x_ref[:, 0], x_ref[:, 1], x_ref[:, 2], "--", label="ref")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D trajectory")

    # make axes roughly equal scale
    xs, ys, zs = X_opt[:, 0], X_opt[:, 1], X_opt[:, 2]
    xr, yr, zr = x_ref[:, 0], x_ref[:, 1], x_ref[:, 2]
    allx = np.concatenate([xs, xr]);
    ally = np.concatenate([ys, yr]);
    allz = np.concatenate([zs, zr])
    xrange = allx.max() - allx.min()
    yrange = ally.max() - ally.min()
    zrange = allz.max() - allz.min()
    max_range = max(xrange, yrange, zrange)
    xmid, ymid, zmid = allx.mean(), ally.mean(), allz.mean()
    ax.set_xlim(xmid - max_range / 2, xmid + max_range / 2)
    ax.set_ylim(ymid - max_range / 2, ymid + max_range / 2)
    ax.set_zlim(zmid - max_range / 2, zmid + max_range / 2)

    ax.legend()
    plt.tight_layout()
    plt.show()
