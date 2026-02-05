import numpy as np

# ------------------------
# Model parameters (from the page)
# ------------------------
m1, m2 = 1.4, 1.0          # [kg]
l1, l2 = 0.30, 0.33        # [m]
s1, s2 = 0.11, 0.16        # COM distances [m]
I1, I2 = 0.025, 0.045      # link inertias [kg m^2]

# viscous friction matrix B
b11 = b22 = 0.05
b12 = b21 = 0.025
Bmat = np.array([[b11, b12],
                 [b21, b22]], dtype=float)

# convenient condensed constants
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
    M11 = a1 + 2*a2*c2
    M12 = a3 + a2*c2
    M22 = a3
    return np.array([[M11, M12],
                     [M12, M22]])

def h_vector(th, thd):
    # Standard 2-link Coriolis/centripetal vector: h = [ -2h*th1d*th2d - h*th2d^2 ; h*th1d^2 ],
    # where h = a2 * sin(th2)
    th1, th2 = th
    th1d, th2d = thd
    h = a2 * np.sin(th2)
    c1 = -2*h*th1d*th2d - h*(th2d**2)
    c2 =  h*(th1d**2)
    return np.array([c1, c2])

def f_continuous(x, u):
    """x=[th1 th2 th1d th2d], u=[tau1 tau2] -> xdot."""
    th = x[:2]
    thd = x[2:]
    M = M_matrix(th)
    rhs = u - h_vector(th, thd) - Bmat @ thd
    thdd = np.linalg.solve(M, rhs)
    return np.hstack([thd, thdd])

def f_discrete_rk4(x, u, dt):
    """One RK4 step with constant u over [t,t+dt]."""
    k1 = f_continuous(x, u)
    k2 = f_continuous(x + 0.5*dt*k1, u)
    k3 = f_continuous(x + 0.5*dt*k2, u)
    k4 = f_continuous(x + dt*k3, u)
    xn = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    # wrap angles for numerical stability
    xn[0] = wrap_angle(xn[0])
    xn[1] = wrap_angle(xn[1])
    return xn

# ------------------------
# iLQR (finite-difference Jacobians)
# ------------------------
def finite_diff_jacobian(f, x, u, dt, eps=1e-6):
    """Central FD Jacobians of discrete dynamics: fx, fu."""
    n = x.size; m = u.size
    fx = np.zeros((n, n)); fu = np.zeros((n, m))
    f0 = f_discrete_rk4(x, u, dt)
    for i in range(n):
        dx = np.zeros_like(x); dx[i] = eps
        fp = f_discrete_rk4(x + dx, u, dt)
        fm = f_discrete_rk4(x - dx, u, dt)
        fx[:, i] = (fp - fm) / (2*eps)
    for j in range(m):
        du = np.zeros_like(u); du[j] = eps
        fp = f_discrete_rk4(x, u + du, dt)
        fm = f_discrete_rk4(x, u - du, dt)
        fu[:, j] = (fp - fm) / (2*eps)
    return fx, fu, f0

# Quadratic costs (with angle wrapping on θ)
def stage_cost(x, u, x_goal, Q, R):
    xe = x.copy()
    xe[0] = wrap_angle(xe[0] - x_goal[0])
    xe[1] = wrap_angle(xe[1] - x_goal[1])
    xe[2:] = xe[2:] - x_goal[2:]
    return 0.5*(xe @ Q @ xe) + 0.5*(u @ R @ u)

def terminal_cost(x, x_goal, Qf):
    xe = x.copy()
    xe[0] = wrap_angle(xe[0] - x_goal[0])
    xe[1] = wrap_angle(xe[1] - x_goal[1])
    xe[2:] = xe[2:] - x_goal[2:]
    return 0.5*(xe @ Qf @ xe)

def ilqr(x0, U_init, dt, x_goal,
         Q, R, Qf,
         u_min=None, u_max=None,
         max_iter=100, alpha_list=None,
         reg_init=1e-6, reg_factor=10.0, tol=1e-6):
    """
    iLQR for x_{k+1}=f(x_k,u_k)
    """
    if alpha_list is None:
        alpha_list = [1.0, 0.7, 0.5, 0.3, 0.1, 0.03, 0.01]

    N = len(U_init)
    n = x0.size; m = U_init[0].size

    X = np.zeros((N+1, n))
    U = U_init.copy()
    X[0] = x0

    # forward rollout with initial U
    cost = 0.0
    for k in range(N):
        if u_min is not None: U[k] = np.maximum(U[k], u_min)
        if u_max is not None: U[k] = np.minimum(U[k], u_max)
        X[k+1] = f_discrete_rk4(X[k], U[k], dt)
        cost += stage_cost(X[k], U[k], x_goal, Q, R)
    cost += terminal_cost(X[-1], x_goal, Qf)

    reg = reg_init
    for it in range(max_iter):
        # Backward pass
        Vx, Vxx = None, None
        kff = [np.zeros(m) for _ in range(N)]
        K = [np.zeros((m, n)) for _ in range(N)]

        # terminal
        xN = X[-1].copy()
        xeN = xN.copy()
        xeN[0] = wrap_angle(xeN[0] - x_goal[0])
        xeN[1] = wrap_angle(xeN[1] - x_goal[1])
        xeN[2:] = xeN[2:] - x_goal[2:]
        Vx = Qf @ xeN
        Vxx = Qf.copy()

        diverged = False
        for k in reversed(range(N)):
            xk, uk = X[k], U[k]
            fx, fu, _ = finite_diff_jacobian(f_discrete_rk4, xk, uk, dt)

            # Quadratic approximation of cost around (xk,uk)
            # l = 1/2 x^T Q x + 1/2 u^T R u
            # with angle wrapping in gradient: approximate locally by simply using xe
            xe = xk.copy()
            xe[0] = wrap_angle(xe[0] - x_goal[0])
            xe[1] = wrap_angle(xe[1] - x_goal[1])
            xe[2:] = xe[2:] - x_goal[2:]
            lx  = Q @ xe
            lu  = R @ uk
            lxx = Q
            luu = R
            lux = np.zeros((m, n))

            # Q-functions
            Qx  = lx + fx.T @ Vx
            Qu  = lu + fu.T @ Vx
            Qxx = lxx + fx.T @ Vxx @ fx
            Quu = luu + fu.T @ Vxx @ fu
            Qux = lux + fu.T @ Vxx @ fx

            # regularize (LM)
            Quu_reg = Quu + reg * np.eye(m)

            # solve for gains
            try:
                Quu_inv = np.linalg.inv(Quu_reg)
            except np.linalg.LinAlgError:
                diverged = True
                break

            kff[k] = -Quu_inv @ Qu
            K[k]   = -Quu_inv @ Qux

            # value function update
            Vx  = Qx + K[k].T @ Quu @ kff[k] + K[k].T @ Qu + Qux.T @ kff[k]
            Vxx = Qxx + K[k].T @ Quu @ K[k] + K[k].T @ Qux + Qux.T @ K[k]
            # symmetrize for numerical stability
            Vxx = 0.5 * (Vxx + Vxx.T)

        if diverged:
            reg *= reg_factor
            # print(f"[iter {it}] Backward pass diverged. Increasing reg -> {reg:.2e}")
            continue

        # Forward line-search
        accepted = False
        for alpha in alpha_list:
            Xnew = np.zeros_like(X)
            Unew = U.copy()
            Xnew[0] = x0
            cost_new = 0.0
            for k in range(N):
                du = alpha * kff[k] + K[k] @ (Xnew[k] - X[k])
                uk = U[k] + du
                if u_min is not None: uk = np.maximum(uk, u_min)
                if u_max is not None: uk = np.minimum(uk, u_max)
                Unew[k] = uk
                Xnew[k+1] = f_discrete_rk4(Xnew[k], uk, dt)
                cost_new += stage_cost(Xnew[k], uk, x_goal, Q, R)
            cost_new += terminal_cost(Xnew[-1], x_goal, Qf)

            if cost_new < cost:
                X, U, cost = Xnew, Unew, cost_new
                reg = max(reg / reg_factor, 1e-12)
                accepted = True
                break

        # print(f"[iter {it}] cost = {cost:.6f}, reg = {reg:.1e}, accepted={accepted}")
        if not accepted:
            reg *= reg_factor
        if abs(cost - (cost_new if accepted else cost)) < tol:
            # print(f"Converged at iter {it}")
            break

    return X, U, cost

# ------------------------
# Demo: swing-up then stabilize
# ------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    dt = 0.01
    T  = 3.0            # horizon seconds (increase if needed)
    N  = int(T/dt)

    # initial and goal
    x0 = np.array([-np.pi/2, 0.0, 0.0, 0.0])         # both links "down", no velocity
    x_goal = np.array([np.pi/2, 0.0, 0.0, 0.0]) # shoulder 90deg, elbow straight

    # costs
    Q  = np.diag([40.0, 20.0, 1.0, 1.0])
    R  = np.diag([0.02, 0.02])
    Qf = np.diag([400.0, 200.0, 10.0, 10.0])

    # initial guess: zero torques or small randoms help exploration
    U_init = [np.zeros(2) for _ in range(N)]

    # torque limits (optional)
    tau_max = np.array([8.0, 6.0])
    u_min = -tau_max
    u_max =  tau_max

    Xopt, Uopt, J = ilqr(x0, U_init, dt, x_goal, Q, R, Qf,
                         u_min=u_min, u_max=u_max,
                         max_iter=200)

    print("Final cost:", J)
    print("Final state:", Xopt[-1])

    # ---- quick plots (uncomment if you want) ----
    import matplotlib.pyplot as plt
    t = np.arange(N+1)*dt
    Xp = np.array(Xopt); Up = np.array(Uopt)
    fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axs[0].plot(t, Xp[:,0], label="θ1"); axs[0].plot(t, Xp[:,1], label="θ2")
    axs[0].axhline(x_goal[0], ls="--"); axs[0].axhline(x_goal[1], ls="--")
    axs[0].set_ylabel("angle [rad]"); axs[0].legend()
    axs[1].plot(t, Xp[:,2], label="θ̇1"); axs[1].plot(t, Xp[:,3], label="θ̇2")
    axs[1].set_ylabel("angular rate [rad/s]"); axs[1].legend()
    axs[2].plot(t[:-1], Up[:,0], label="τ1"); axs[2].plot(t[:-1], Up[:,1], label="τ2")
    axs[2].set_ylabel("torque [Nm]"); axs[2].set_xlabel("time [s]"); axs[2].legend()
    plt.tight_layout(); plt.show()

    # ------------------------
    # Animation of 2-link arm
    # ------------------------
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation


    def animate_arm(X, l1, l2, dt, save=False, fname="arm_ilqr.mp4"):
        th1 = X[:, 0]
        th2 = X[:, 1]
        x0, y0 = 0, 0
        x1 = l1 * np.cos(th1)
        y1 = l1 * np.sin(th1)
        x2 = x1 + l2 * np.cos(th1 + th2)
        y2 = y1 + l2 * np.sin(th1 + th2)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-(l1 + l2) * 1.1, (l1 + l2) * 1.1)
        ax.set_ylim(-(l1 + l2) * 1.1, (l1 + l2) * 1.1)
        ax.set_aspect("equal")
        ax.grid(True)
        line, = ax.plot([], [], 'o-', lw=3, color='tab:blue')
        trace, = ax.plot([], [], 'r--', lw=1, alpha=0.4)
        tip_path_x, tip_path_y = [], []

        def init():
            line.set_data([], [])
            trace.set_data([], [])
            return line, trace

        def update(frame):
            xdata = [x0, x1[frame], x2[frame]]
            ydata = [y0, y1[frame], y2[frame]]
            line.set_data(xdata, ydata)
            tip_path_x.append(x2[frame])
            tip_path_y.append(y2[frame])
            trace.set_data(tip_path_x, tip_path_y)
            return line, trace

        ani = FuncAnimation(fig, update, frames=len(th1),
                            init_func=init, interval=dt * 1000, blit=True)
        if save:
            ani.save(fname, fps=int(1 / dt), dpi=150)
            print(f"Animation saved to {fname}")
        else:
            plt.show()


    # Run the animation
    animate_arm(Xopt, l1, l2, dt, save=False)

