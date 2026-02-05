#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iLQR for a 2-link arm (horizontal plane) with ONLY the shoulder (joint 1) actuated.
Elbow torque is fixed to zero. Includes animation.

Model is taken from the provided page:
M(theta) * theta_dd + C(theta, theta_d) + B * theta_d = tau
with:
M = [[a1 + 2 a2 cos(th2), a3 + a2 cos(th2)],
     [a3 + a2 cos(th2),   a3]]
C vector (Coriolis/centripetal) implemented in the standard 2-link form:
    h = a2 sin(th2)
    c1 = -2 h th1d th2d - h th2d^2
    c2 =  h th1d^2
tau = [tau1, 0]^T (underactuated)

Author: you + ChatGPT
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
# ------------------------
# Parameters from the page
# ------------------------
m1, m2 = 1.4, 1.0          # masses [kg]
l1, l2 = 0.30, 0.33        # link lengths [m]
s1, s2 = 0.11, 0.16        # COM distances [m]
I1, I2 = 0.025, 0.045      # inertias [kg*m^2]

# viscous friction matrix B
b11 = b22 = 0.05
b12 = b21 = 0.025
Bmat = np.array([[b11, b12],
                 [b21, b22]], dtype=float)

# condensed constants
a1 = I1 + I2 + m2 * (l1**2)
a2 = m2 * l1 * s2
a3 = I2


def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2*np.pi) - np.pi


# ------------------------
# Dynamics
# ------------------------
def M_matrix(th):
    th1, th2 = th
    c2 = np.cos(th2)
    M11 = a1 + 2.0 * a2 * c2
    M12 = a3 + a2 * c2
    M22 = a3
    return np.array([[M11, M12],
                     [M12, M22]], dtype=float)


def h_vector(th, thd):
    """Coriolis/centripetal vector in standard 2-link form."""
    th1, th2 = th
    th1d, th2d = thd
    h = a2 * np.sin(th2)
    c1 = -2.0 * h * th1d * th2d - h * (th2d ** 2)
    c2 =  h * (th1d ** 2)
    return np.array([c1, c2], dtype=float)


def f_continuous_under(x, u1):
    """
    Continuous-time dynamics with only shoulder torque.
    x = [th1, th2, th1d, th2d], u1 = scalar shoulder torque.
    """
    th = x[:2]
    thd = x[2:]
    tau = np.array([float(u1), 0.0], dtype=float)
    M = M_matrix(th)
    rhs = tau - h_vector(th, thd) - Bmat @ thd
    thdd = np.linalg.solve(M, rhs)
    return np.hstack([thd, thdd])


def f_discrete_under(x, u1, dt):
    """RK4 step with constant u1 on [t, t+dt]."""
    k1 = f_continuous_under(x, u1)
    k2 = f_continuous_under(x + 0.5 * dt * k1, u1)
    k3 = f_continuous_under(x + 0.5 * dt * k2, u1)
    k4 = f_continuous_under(x + dt * k3, u1)
    xn = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    xn[0] = wrap_angle(xn[0])
    xn[1] = wrap_angle(xn[1])
    return xn


def fd_jac_under(x, u1, dt, eps=1e-6):
    """
    Central finite-difference Jacobians for the discrete map x+ = f(x,u1).
    Returns fx (n,n), fu (n,1), and f0 (n,)
    """
    n = x.size
    f0 = f_discrete_under(x, u1, dt)
    fx = np.zeros((n, n), dtype=float)

    # df/dx
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        fp = f_discrete_under(x + dx, u1, dt)
        fm = f_discrete_under(x - dx, u1, dt)
        fx[:, i] = (fp - fm) / (2.0 * eps)

    # df/du
    fp = f_discrete_under(x, u1 + eps, dt)
    fm = f_discrete_under(x, u1 - eps, dt)
    fu = ((fp - fm) / (2.0 * eps)).reshape(n, 1)
    return fx, fu, f0


# ------------------------
# Costs
# ------------------------
def angle_state_error(x, x_goal):
    """Angle-aware state error (wrap angle diffs only)."""
    xe = x.copy()
    xe[0] = wrap_angle(xe[0] - x_goal[0])
    xe[1] = wrap_angle(xe[1] - x_goal[1])
    xe[2:] = xe[2:] - x_goal[2:]
    return xe


def stage_cost(x, u1, x_goal, Q, R_scalar):
    xe = angle_state_error(x, x_goal)
    return 0.5 * (xe @ Q @ xe) + 0.5 * (R_scalar * (float(u1) ** 2))


def terminal_cost(x, x_goal, Qf):
    xe = angle_state_error(x, x_goal)
    return 0.5 * (xe @ Qf @ xe)


# ------------------------
# iLQR (m=1)
# ------------------------
def ilqr_under(x0, U_init, dt, x_goal,
               Q, R_scalar, Qf,
               u_min=None, u_max=None,
               max_iter=150,
               alpha_list=None,
               reg_init=1e-6, reg_factor=10.0,
               tol=1e-7):
    """
    iLQR for underactuated case with scalar control u1.

    Q, Qf : (n,n)
    R_scalar : float
    U_init : shape (N,) array-like of scalars
    """
    if alpha_list is None:
        alpha_list = [1.0, 0.7, 0.5, 0.3, 0.1, 0.03, 0.01]

    U = np.asarray(U_init, dtype=float).copy()
    N = U.shape[0]
    n = x0.size

    # rollout
    X = np.zeros((N + 1, n), dtype=float)
    X[0] = x0
    J = 0.0
    for k in range(N):
        uk = float(U[k])
        if u_min is not None: uk = max(uk, float(u_min))
        if u_max is not None: uk = min(uk, float(u_max))
        U[k] = uk
        X[k + 1] = f_discrete_under(X[k], uk, dt)
        J += stage_cost(X[k], uk, x_goal, Q, R_scalar)
    J += terminal_cost(X[-1], x_goal, Qf)

    reg = reg_init

    for it in range(max_iter):
        # Terminal value function
        xeN = angle_state_error(X[-1], x_goal)
        Vx = Qf @ xeN                      # (n,)
        Vxx = Qf.copy()                    # (n,n)

        kff = np.zeros(N, dtype=float)     # feedforward scalars
        K = np.zeros((N, n), dtype=float)  # feedback gains (1xn)

        diverged = False

        # ---------- Backward pass ----------
        for k in reversed(range(N)):
            xk = X[k]
            uk = float(U[k])

            fx, fu, _ = fd_jac_under(xk, uk, dt)  # fu: (n,1)

            # Quadratic approximation of stage cost l
            xe = angle_state_error(xk, x_goal)
            lx = Q @ xe                        # (n,)
            lu = R_scalar * uk                 # scalar
            lxx = Q
            luu = R_scalar                     # scalar
            lux = np.zeros((1, n), dtype=float)

            # Q-function expansions
            Qx  = lx + fx.T @ Vx                                # (n,)
            Qu  = lu + float((fu.T @ Vx).item())                # scalar
            Qxx = lxx + fx.T @ Vxx @ fx                         # (n,n)
            Quu = luu + float((fu.T @ Vxx @ fu).item())         # scalar
            Qux = lux + (fu.T @ Vxx @ fx)                       # (1,n)

            # Regularize (LM)
            Quu_reg = Quu + reg

            if Quu_reg <= 0:
                # make it positive definite
                reg *= reg_factor
                diverged = True
                break

            # Gains
            kff_k = - Quu_reg**-1 * Qu                      # scalar
            K_k   = - (Qux / Quu_reg)                       # (1,n) -> row

            kff[k] = float(kff_k)
            K[k, :] = K_k.reshape(-1)

            # Value function update
            # Vx = Qx + K^T * Quu * kff + K^T * Qu + Qux^T * kff
            term1 = Qx
            term2 = (K_k.T * (Quu * kff_k)).reshape(-1)
            term3 = (K_k.T * Qu).reshape(-1)
            term4 = (Qux.T.reshape(n, 1) * kff_k).reshape(-1)
            Vx = term1 + term2 + term3 + term4

            # Vxx = Qxx + K^T*Quu*K + K^T*Qux + Qux^T*K
            Vxx = Qxx + (K_k.T @ (Quu * K_k)) + (K_k.T @ Qux) + (Qux.T @ K_k)
            Vxx = 0.5 * (Vxx + Vxx.T)  # symmetrize

        if diverged:
            # try again with larger regularization
            # print(f"[it {it}] backward diverged, reg -> {reg:.1e}")
            continue

        # ---------- Forward pass with line search ----------
        accepted = False
        best_cost = None
        best_X = None
        best_U = None

        for alpha in alpha_list:
            Xnew = np.zeros_like(X)
            Unew = U.copy()
            Xnew[0] = x0
            Jnew = 0.0

            for k in range(N):
                du = alpha * kff[k] + float(K[k] @ (Xnew[k] - X[k]))
                uk = U[k] + du
                if u_min is not None: uk = max(uk, float(u_min))
                if u_max is not None: uk = min(uk, float(u_max))
                Unew[k] = uk
                Xnew[k + 1] = f_discrete_under(Xnew[k], uk, dt)
                Jnew += stage_cost(Xnew[k], uk, x_goal, Q, R_scalar)
            Jnew += terminal_cost(Xnew[-1], x_goal, Qf)

            if (best_cost is None) or (Jnew < best_cost):
                best_cost, best_X, best_U = Jnew, Xnew, Unew

            if Jnew < J:
                X, U, J = Xnew, Unew, Jnew
                reg = max(reg / reg_factor, 1e-12)
                accepted = True
                break

        if not accepted:
            reg *= reg_factor
            # If even the best rollout didn't improve much, stop
            if best_cost is not None and abs(best_cost - J) < tol:
                X, U, J = best_X, best_U, best_cost
                break

        # Convergence check
        if abs(J - best_cost) < tol:
            break

    return X, U, J


# ------------------------
# Animation
# ------------------------
def animate_arm(X, l1, l2, dt, save=False, fname="arm_ilqr_under.mp4"):

    th1 = X[:, 0]
    th2 = X[:, 1]

    x0 = np.zeros_like(th1)
    y0 = np.zeros_like(th1)
    x1 = l1 * np.cos(th1)
    y1 = l1 * np.sin(th1)
    x2 = x1 + l2 * np.cos(th1 + th2)
    y2 = y1 + l2 * np.sin(th1 + th2)

    Ltot = l1 + l2
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1.1 * Ltot, 1.1 * Ltot)
    ax.set_ylim(-1.1 * Ltot, 1.1 * Ltot)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    line, = ax.plot([], [], "o-", lw=3)
    trace, = ax.plot([], [], "--", lw=1, alpha=0.4)
    tip_x, tip_y = [], []

    def init():
        line.set_data([], [])
        trace.set_data([], [])
        return line, trace

    def update(i):
        line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        tip_x.append(x2[i]); tip_y.append(y2[i])
        trace.set_data(tip_x, tip_y)
        return line, trace

    ani = FuncAnimation(fig, update, frames=len(th1),
                        init_func=init, interval=max(1, int(dt*1000)),
                        blit=True)
    if save:
        ani.save(fname, fps=int(1.0/dt), dpi=150)
        print(f"Animation saved to {fname}")
    else:
        plt.show()

# ------------------------
# Main demo
# ------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    # Discretization and horizon
    dt = 0.01
    T = 4    # longer horizon helps underactuated case
    N = int(T / dt)

    # Start "down", zero velocity (choose per your angle convention)
    x0 = np.array([-np.pi/2, 0.0, 0.0, 0.0])

    # Goal pose (example): shoulder up 90 deg, elbow straight
    x_goal = np.array([np.pi/2, 0.0, 0.0, 0.0])

    # Costs
    Q  = 10*np.diag([40.0, 20.0, 1.0, 1.0])
    Qf = 10*np.diag([400.0, 200.0, 10.0, 10.0])
    R_scalar = 0.02

    # Control limits (only shoulder torque)
    tau1_max = 8.0
    u_min, u_max = -tau1_max, tau1_max

    # Initial control guess
    # U_init = np.zeros(N, dtype=float)  # can add small noise to help exploration
    # initial control guess
    # U_init = np.zeros(N)
    # # option 1: random jitter
    # U_init += 0.05 * np.random.randn(N)
    # # option 2: hand-crafted torque pulse
    # pulse_steps = int(0.25 / dt)
    # U_init[:pulse_steps] += 3.0
    # U_init[pulse_steps:2 * pulse_steps] -= 2.0

    U_init = 0.05 * np.random.randn(N)  # small jitter
    pulse_T = 0.5 # duration of excitation (s)
    pulse_steps = 30 #int(pulse_T / dt)
    freq = 0.2 * np.pi / pulse_T  # one oscillation in pulse_T
    amp = 4.0  # Nm amplitude

    # generate decaying sinusoid over the first pulse_T seconds
    for k in range(pulse_steps):
        t = k * dt
        U_init[k] += amp * np.sin(freq * t) * np.exp(-2.0 * t)

    # Solve with iLQR
    Xopt, Uopt, J = ilqr_under(x0, U_init, dt, x_goal,
                               Q, R_scalar, Qf,
                               u_min=u_min, u_max=u_max,
                               max_iter=200)

    print("Final cost:", J)
    print("Final state:", Xopt[-1])
    animate_arm(Xopt,l1,l2,dt)



