import numpy as np
#https://github.com/Bharath2/iLQR/tree/main

def dynamics(x, u, params):
    """
    Nonlinear dynamics of double inverted pendulum on a cart
    x = [x_cart, theta1, theta2, x_dot, theta1_dot, theta2_dot]
    u = scalar control force on the cart
    """
    m0, m1, m2, L1, L2, g = params['m0'], params['m1'], params['m2'], params['L1'], params['L2'], params['g']

    # Unpack states
    x_cart, th1, th2, x_dot, th1_dot, th2_dot = x

    # Equations from dynamics (simplified for illustration):
    # Using Lagrangian or equations from paper; for brevity, approximate here:
    # You will need to implement full dynamic equations (I provide a simplified version)

    # Auxiliary variables for the equations
    c1 = np.cos(th1)
    s1 = np.sin(th1)
    c2 = np.cos(th2)
    s2 = np.sin(th2)

    # Mass matrix D (3x3), Coriolis C, gravity G
    # Build matrices per your model (from your MATLAB d0, dg, etc.)

    # For now, placeholder simple linearization around upright for demonstration:
    D = np.eye(3)
    C = np.zeros((3,3))
    G = np.array([0, -m1 * g * L1/2 * s1, -m2 * g * L2/2 * s2])

    # State derivatives
    q_dot = np.array([x_dot, th1_dot, th2_dot])
    # Compute accelerations (placeholder):
    q_ddot = np.linalg.inv(D) @ (np.array([u, 0, 0]) - C @ q_dot - G)

    # Compose state derivative
    x_dot_vec = np.zeros(6)
    x_dot_vec[0:3] = q_dot
    x_dot_vec[3:6] = q_ddot

    return x_dot_vec

def cost(x, u, x_goal, Q, R):
    dx = x - x_goal
    return dx.T @ Q @ dx + u.T @ R @ u

def finite_difference_jacobian(f, x, u, params, eps=1e-5):
    n = x.size
    m = u.size if hasattr(u, 'size') else 1
    A = np.zeros((n, n))
    B = np.zeros((n, m))
    fx = f(x, u, params)

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        A[:, i] = (f(x + dx, u, params) - fx) / eps

    for i in range(m):
        du = np.zeros(m)
        du[i] = eps
        B[:, i] = (f(x, u + du, params) - fx) / eps

    return A, B

def ilqr(f, x0, u_init, params, Q, R, Qf, x_goal, max_iter=100, tol=1e-6, dt=0.02):
    n = x0.size
    m = u_init.shape[1]
    N = u_init.shape[0]

    u = u_init.copy()
    x = np.zeros((N+1, n))
    x[0] = x0

    for iteration in range(max_iter):
        # Forward rollout with current control
        for k in range(N):
            x_dot = f(x[k], u[k], params)
            x[k+1] = x[k] + dt * x_dot

        # Compute cost-to-go derivatives backward
        Vx = Qf @ (x[-1] - x_goal)
        Vxx = Qf

        k_seq = np.zeros((N, m))
        K_seq = np.zeros((N, m, n))

        for k in reversed(range(N)):
            A, B = finite_difference_jacobian(f, x[k], u[k], params, eps=1e-5)

            Qx = Q @ (x[k] - x_goal) + A.T @ Vx
            Qu = R @ u[k] + B.T @ Vx
            Qxx = Q + A.T @ Vxx @ A
            Quu = R + B.T @ Vxx @ B
            Qux = B.T @ Vxx @ A

            # Regularize Quu
            Quu_reg = Quu + 1e-6 * np.eye(m)

            # Compute gains
            K = -np.linalg.solve(Quu_reg, Qux)
            k_ff = -np.linalg.solve(Quu_reg, Qu)

            k_seq[k] = k_ff
            K_seq[k] = K

            Vx = Qx + K.T @ Quu @ k_ff + K.T @ Qu + Qux.T @ k_ff
            Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K

        # Line search and control update
        alpha = 1.0
        cost_prev = np.sum([cost(x[k], u[k], x_goal, Q, R) for k in range(N)]) + (x[-1] - x_goal).T @ Qf @ (x[-1] - x_goal)
        while alpha > 1e-4:
            x_new = np.zeros_like(x)
            x_new[0] = x0
            u_new = np.zeros_like(u)
            for k in range(N):
                u_new[k] = u[k] + alpha * k_seq[k] + K_seq[k] @ (x_new[k] - x[k])
                x_dot = f(x_new[k], u_new[k], params)
                x_new[k+1] = x_new[k] + dt * x_dot

            cost_new = np.sum([cost(x_new[k], u_new[k], x_goal, Q, R) for k in range(N)]) + (x_new[-1] - x_goal).T @ Qf @ (x_new[-1] - x_goal)

            if cost_new < cost_prev:
                break
            alpha *= 0.5

        if np.abs(cost_prev - cost_new) < tol:
            break

        u = u_new
        x = x_new

    return x, u

# System parameters
params = {
    'm0': 1.5,
    'm1': 0.5,
    'm2': 0.75,
    'L1': 0.5,
    'L2': 0.75,
    'g': 10,
}

dt = 0.02
N = 300  # horizon length
n = 6    # state dimension
m = 1    # control dimension

# Initial state (cart pos, lower angle, upper angle, velocities)
x0 = np.array([0, np.pi, np.pi, 0, 0, 0])  # pendulum hanging down (pi radians)

# Goal state (upright)
x_goal = np.array([0, 0, 0, 0, 0, 0])

# Initial control guess
u_init = np.zeros((N, m))

# Cost matrices
Q = np.diag([1, 10, 10, 1, 1, 1])
R = np.diag([0.01])
Qf = np.diag([100, 100, 100, 10, 10, 10])

# Run iLQR
x_traj, u_traj = ilqr(dynamics, x0, u_init, params, Q, R, Qf, x_goal, max_iter=50, dt=dt)

