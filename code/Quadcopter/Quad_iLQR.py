import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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



class QuadcopterEnv():
    """
    This environment describes a fully nonlinear quadcopter

    ## Action Space
    | Num | Action                 | Min | Max | Name | Unit |
    | --- | ---------------------- | --- | --- | ---- | ---- |
    | 0   | Torque on first rotor  | -1  |  1  | w0d  |  N   |
    | 1   | Torque on second rotor | -1  |  1  | w1d  |  N   |
    | 2   | Torque on third rotor  | -1  |  1  | w2d  |  N   |
    | 3   | Torque on fourth rotor | -1  |  1  | w3d  |  N   |

    ## Observation Space
    | Num | Observation                            | Min  | Max | Name | Unit  |
    | --- | -------------------------------------- | ---- | --- | ---- | ----- |
    | 0   | x-coordinate of the center of mass     | -Inf | Inf |  x   |  m    |
    | 1   | y-coordinate of the center of mass     | -Inf | Inf |  y   |  m    |
    | 2   | z-coordinate of the center of mass     | -Inf | Inf |  z   |  m    |
    | 3   | w-orientaiont of the body (quaternion) | -Inf | Inf |  q0  |  rad  |
    | 4   | x-orientaiont of the body (quaternion) | -Inf | Inf |  q1  |  rad  |
    | 5   | y-orientaiont of the body (quaternion) | -Inf | Inf |  q2  |  rad  |
    | 6   | z-orientaiont of the body (quaternion) | -Inf | Inf |  q3  |  rad  |

    | 7   | x-velocity of the center of mass       | -Inf | Inf |  xd  |  m/s  |
    | 8   | y-velocity of the center of mass       | -Inf | Inf |  yd  |  m/s  |
    | 9   | z-velocity of the center of mass       | -Inf | Inf |  zd  |  m/s  |
    | 10  | x-angular velocity of the body         | -Inf | Inf |  p   | rad/s |
    | 11  | y-angular velocity of the body         | -Inf | Inf |  q   | rad/s |
    | 12  | z-angular velocity of the body         | -Inf | Inf |  r   | rad/s |
    | 13  | angular velocity of the first rotor    | -Inf | Inf |  w0  | rad/s |
    | 14  | angular velocity of the second rotor   | -Inf | Inf |  w1  | rad/s |
    | 15  | angular velocity of the third rotor    | -Inf | Inf |  w2  | rad/s |
    | 16  | angular velocity of the fourth rotor   | -Inf | Inf |  w3  | rad/s |


    ## Starting State
    All observations start from hover with a Gaussian noise of magnitude `reset_noise_scale'

    ## Episode End
    1. Any of the states goes out of bounds
    2. The Quadcopter collides with one of the cylinder obstacles

    NOTES:
    John integrated the proportional control of the rotors directly into the
    equations of motion to more accurately reflect the closed loop system
    we will be controlling with a second outer loop. This inner loop is akin
    to the ESC which will be onboard many quadcopters which directly controls
    the rotor speeds to be what is commanded.
    """

    def __init__(self, reset_noise_scale: float = 1e-2, dt: float = 0.01,
                 cylinder_radii=[0.7, 0.7]):

        self.name = "Quadcopter"
        self.state_size = 17
        self.action_size = 4
        self.action_min = np.array([[-1., -1., -1., -1.]])
        self.action_max = np.array([[1., 1., 1., 1.]])
        self.position_states = [0, 1, 2, 3, 4, 5, 6]
        self.velocity_states = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        ### Obstacles: cylinders along the z-axis
        self.target_position = np.array([7., 0., 0.])
        self.N_cylinders = 2  # 2 cylinders
        self.cylinder_radii = cylinder_radii  # radius of the cylinders
        self.cylinder_xc = [2.5, 5.2]  # cylinders x center position
        self.cylinder_yc = [0.5, -0.5]  # cylinders y center position

        ### Fundamental quad parameters
        self.g = 9.81  # gravity (m/s^2)
        self.mB = 1.2  # mass (kg)
        self.dxm = 0.16  # arm length (m)
        self.dym = 0.16  # arm length (m)
        self.dzm = 0.01  # arm height (m)
        self.IB = np.array([[0.0123, 0, 0],
                            [0, 0.0123, 0],
                            [0, 0, 0.0224]])  # Inertial tensor (kg*m^2)
        self.IRzz = 2.7e-5  # rotor moment of inertia (kg*m^2)
        self.Cd = 0.1  # drag coefficient (omnidirectional)
        self.kTh = 1.076e-5  # thrust coeff (N/(rad/s)^2)  (1.18e-7 N/RPM^2)
        self.kTo = 1.632e-7  # torque coeff (Nm/(rad/s)^2)  (1.79e-9 Nm/RPM^2)
        self.minThr = 0.1 * 4  # Minimum total thrust (N)
        self.maxThr = 9.18 * 4  # Maximum total thrust (N)
        self.minWmotor = 75  # Minimum motor rotation speed (rad/s)
        self.maxWmotor = 925  # Maximum motor rotation speed (rad/s)
        self.tau = 0.015  # Value for second order system for Motor dynamics
        self.kp = 1.0  # Value for second order system for Motor dynamics
        self.damp = 1.0  # Value for second order system for Motor dynamics
        self.usePrecession = True  # model precession or not
        self.w_hover = 522.9847140714692  # hardcoded hover rotor speed (rad/s)

        ### post init useful parameters for quad
        self.B0 = np.array([[self.kTh, self.kTh, self.kTh, self.kTh],
                            [self.dym * self.kTh, - self.dym * self.kTh, -self.dym * self.kTh, self.dym * self.kTh],
                            [self.dxm * self.kTh, self.dxm * self.kTh, -self.dxm * self.kTh, -self.dxm * self.kTh],
                            [-self.kTo, self.kTo, - self.kTo, self.kTo]])  # actuation matrix

        self.low_bound = np.array([-100, -100, -100, *[-np.inf] * 4, *[-100] * 3, *[-100] * 3, *[self.minWmotor] * 4])
        # xyz       q0123         xdydzd    pdqdrd    w0123

        self.high_bound = np.array([100, 100, 100, *[np.inf] * 4, *[100] * 3, *[100] * 3, *[self.maxWmotor] * 4])
        # xyz      q0123        xdydzd   pdqdrd   w0123

        self.dt = dt  # time step
        self.reset_noise = reset_noise_scale
        self.hover_state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, *[522.9847140714692] * 4])

    def reset(self):
        self.state = self.hover_state.copy() + self.reset_noise * np.random.randn(self.state_size)
        return self.state.copy()

    def reset_to(self, state):
        # assert all(state > self.x_lb) and all(state < self.x_ub), "state is out of bounds x_lb, x_ub"
        self.state = state.copy()
        return state

    def is_in_obstacle(self, x, y):
        """Checks whether (x,y) position is inside an obstacle"""
        for i in range(self.N_cylinders):
            inside = self.cylinder_radii[i] ** 2 >= (x - self.cylinder_xc[i]) ** 2 + (y - self.cylinder_yc[i]) ** 2
            if inside:
                return True
        return False

    def step(self, u):

        q0 = self.state[3]
        q1 = self.state[4]
        q2 = self.state[5]
        q3 = self.state[6]
        xdot = self.state[7]
        ydot = self.state[8]
        zdot = self.state[9]
        p = self.state[10]
        q = self.state[11]
        r = self.state[12]
        wM1 = self.state[13]
        wM2 = self.state[14]
        wM3 = self.state[15]
        wM4 = self.state[16]

        # instantaneous thrusts and torques generated by the current w0...w3
        wMotor = np.stack([wM1, wM2, wM3, wM4])
        wMotor = np.clip(wMotor, self.minWmotor, self.maxWmotor)  # this clip shouldn't occur within the dynamics
        th = self.kTh * wMotor ** 2  # thrust
        to = self.kTo * wMotor ** 2  # torque

        # state derivates (from sympy.mechanics derivation)
        xd = np.stack(
            [
                xdot,
                ydot,
                zdot,
                -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
                0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
                0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
                -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
                (self.Cd * np.sign(-xdot) * xdot ** 2
                 - 2 * (q0 * q2 + q1 * q3) * (th[0] + th[1] + th[2] + th[3])
                 )
                / self.mB,  # xdd
                (
                        self.Cd * np.sign(-ydot) * ydot ** 2
                        + 2 * (q0 * q1 - q2 * q3) * (th[0] + th[1] + th[2] + th[3])
                )
                / self.mB,  # ydd
                (
                        - self.Cd * np.sign(zdot) * zdot ** 2
                        - (th[0] + th[1] + th[2] + th[3])
                        * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                        + self.g * self.mB
                )
                / self.mB,  # zdd (the - in front turns increased height to be positive - SWU)
                (
                        (self.IB[1, 1] - self.IB[2, 2]) * q * r
                        - self.usePrecession * self.IRzz * (wM1 - wM2 + wM3 - wM4) * q
                        + (th[0] - th[1] - th[2] + th[3]) * self.dym
                )
                / self.IB[0, 0],  # pd
                (
                        (self.IB[2, 2] - self.IB[0, 0]) * p * r
                        + self.usePrecession * self.IRzz * (wM1 - wM2 + wM3 - wM4) * p
                        + (th[0] + th[1] - th[2] - th[3]) * self.dxm
                )
                / self.IB[1, 1],  # qd
                ((self.IB[0, 0] - self.IB[1, 1]) * p * q - to[0] + to[1] - to[2] + to[3]) / self.IB[2, 2],  # rd
                u[0] / self.IRzz, u[1] / self.IRzz, u[2] / self.IRzz, u[3] / self.IRzz  # w0d ... w3d
            ]
        )

        self.state += xd * self.dt  # one time step forward
        # Clip the rotor speeds within limits
        self.state[13:17] = np.clip(self.state[13:17], self.low_bound[13:17], self.high_bound[13:17])

        out_of_bound = any(self.state < self.low_bound) or any(self.state > self.high_bound)  # out of bound state
        collided = self.is_in_obstacle(self.state[0], self.state[1])
        distance_to_target = np.linalg.norm(self.state[:3] - self.target_position)
        reward = 1 - out_of_bound - collided - distance_to_target
        terminated = out_of_bound or collided

        return self.state.copy(), reward, terminated, False, None

# ---------- utilities ----------
def quat_normalize(q):
    q = q.copy()
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1., 0., 0., 0.])
    return q / n

def clamp(u, umin, umax):
    return np.minimum(np.maximum(u, umin), umax)

# ---------- pure discrete dynamics cloned from your env.step (Euler) ----------
def f_discrete(x, u, dt, params):
    # unpack params
    g = params["g"]; mB = params["mB"]; dym = params["dym"]; dxm = params["dxm"]
    IB = params["IB"]; IRzz = params["IRzz"]; Cd = params["Cd"]
    kTh = params["kTh"]; kTo = params["kTo"]
    usePrecession = params["usePrecession"]
    minWmotor = params["minWmotor"]; maxWmotor = params["maxWmotor"]

    # state aliases
    x_next = x.copy()
    q0,q1,q2,q3 = x[3:7]
    xdot,ydot,zdot = x[7:10]
    p,q,r = x[10:13]
    wM1,wM2,wM3,wM4 = x[13:17]

    # rotor thrusts/torques from current rotor speeds
    wMotor = np.array([wM1,wM2,wM3,wM4])
    wMotor = np.clip(wMotor, minWmotor, maxWmotor)
    th = kTh * wMotor**2
    to = kTo * wMotor**2

    # derivatives (same as env.step)
    xd = np.stack([
        xdot,
        ydot,
        zdot,
        -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
         0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
         0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
        -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
        ( Cd * np.sign(-xdot) * xdot**2 - 2*(q0*q2 + q1*q3) * np.sum(th) ) / mB,
        ( Cd * np.sign(-ydot) * ydot**2 + 2*(q0*q1 - q2*q3) * np.sum(th) ) / mB,
        ( -Cd*np.sign(zdot)*zdot**2 - np.sum(th) * (q0**2 - q1**2 - q2**2 + q3**2) + g*mB ) / mB,
        ( (IB[1,1]-IB[2,2]) * q * r - usePrecession*IRzz*(wM1 - wM2 + wM3 - wM4)*q + (th[0]-th[1]-th[2]+th[3]) * dym ) / IB[0,0],
        ( (IB[2,2]-IB[0,0]) * p * r + usePrecession*IRzz*(wM1 - wM2 + wM3 - wM4)*p + (th[0]+th[1]-th[2]-th[3]) * dxm ) / IB[1,1],
        ( (IB[0,0]-IB[1,1]) * p * q - to[0] + to[1] - to[2] + to[3] ) / IB[2,2],
        u[0]/IRzz, u[1]/IRzz, u[2]/IRzz, u[3]/IRzz
    ])

    x_next += dt * xd
    # clamp motors
    x_next[13:17] = np.clip(x_next[13:17], minWmotor, maxWmotor)
    # renormalize quaternion
    x_next[3:7] = quat_normalize(x_next[3:7])
    return x_next

# ---------- linearization via finite differences ----------
def finite_diff_jacobian(f, x, u, dt, params, eps=1e-5):
    n = x.size; m = u.size
    fx = f(x, u, dt, params)
    A = np.zeros((n, n)); B = np.zeros((n, m))
    for i in range(n):
        dx = np.zeros_like(x); dx[i] = eps
        A[:, i] = (f(x+dx, u, dt, params) - fx) / eps
    for j in range(m):
        du = np.zeros_like(u); du[j] = eps
        B[:, j] = (f(x, u+du, dt, params) - fx) / eps
    return A, B

# ===============================================================
# ========  COST FUNCTIONS (fixed for quaternion + hover)  ======
# ===============================================================

def running_cost(x, u, target, q_hover, w_hover,
                 Qpos, Qquat, Qvel, Qmot, R):
    """
    Quadratic running cost for quadrotor iLQR.
    """
    # State components
    pos  = x[0:3]
    quat = x[3:7]
    linv = x[7:10]
    angv = x[10:13]
    mot  = x[13:17]

    # Errors
    e_pos  = pos  - target
    e_quat = quat - q_hover
    e_linv = linv
    e_angv = angv
    e_mot  = mot  - np.array([w_hover]*4)

    # Weighted quadratic cost
    cost = (
        e_pos  @ Qpos  @ e_pos +
        e_quat @ Qquat @ e_quat +
        e_linv @ Qvel  @ e_linv +
        e_angv @ Qvel  @ e_angv +
        e_mot  @ Qmot  @ e_mot +
        u @ R @ u
    )
    return cost


def terminal_cost(x, target, q_hover, Qf_pos, Qf_quat):
    """
    Terminal cost for quadrotor iLQR.
    """
    pos  = x[0:3]
    quat = x[3:7]
    e_pos  = pos  - target
    e_quat = quat - q_hover
    return e_pos @ Qf_pos @ e_pos + e_quat @ Qf_quat @ e_quat


def cost_derivatives(x, u, target, q_hover, w_hover,
                     Qpos, Qquat, Qvel, Qmot, R):
    """
    Derivatives of running cost wrt x and u (quadratic form).
    """
    nx = x.size
    nu = u.size
    lx = np.zeros(nx)
    lu = np.zeros(nu)
    Lxx = np.zeros((nx, nx))
    Luu = np.zeros((nu, nu))
    Lux = np.zeros((nu, nx))

    # Index slices
    pos  = slice(0, 3)
    quat = slice(3, 7)
    linv = slice(7, 10)
    angv = slice(10, 13)
    mot  = slice(13, 17)

    # Errors
    e_pos  = x[pos]  - target
    e_quat = x[quat] - q_hover
    e_linv = x[linv]
    e_angv = x[angv]
    e_mot  = x[mot]  - np.array([w_hover]*4)

    # Gradients
    lx[pos]  = 2 * Qpos  @ e_pos
    lx[quat] = 2 * Qquat @ e_quat
    lx[linv] = 2 * Qvel  @ e_linv
    lx[angv] = 2 * Qvel  @ e_angv
    lx[mot]  = 2 * Qmot  @ e_mot
    lu       = 2 * R @ u

    # Hessians
    Lxx[pos,  pos]  = 2 * Qpos
    Lxx[quat, quat] = 2 * Qquat
    Lxx[linv, linv] = 2 * Qvel
    Lxx[angv, angv] = 2 * Qvel
    Lxx[mot,  mot]  = 2 * Qmot
    Luu              = 2 * R

    return lx, lu, Lxx, Luu, Lux


def terminal_derivatives(x, target, q_hover, Qf_pos, Qf_quat):
    """
    Derivatives of terminal cost.
    """
    nx = x.size
    Vx = np.zeros(nx)
    Vxx = np.zeros((nx, nx))

    pos  = slice(0, 3)
    quat = slice(3, 7)
    e_pos  = x[pos]  - target
    e_quat = x[quat] - q_hover

    Vx[pos]  = 2 * Qf_pos  @ e_pos
    Vx[quat] = 2 * Qf_quat @ e_quat
    Vxx[pos,  pos]  = 2 * Qf_pos
    Vxx[quat, quat] = 2 * Qf_quat
    return Vx, Vxx


# ---------- iLQR ----------
def ilqr(
    f, x0, U_init, dt, params, # dynamics
    target_pos, w_hover,        # targets
    max_iter=100, tol=1e-3, reg_init=1e-6,
    Qpos_scale=50.0, Qquat_scale=10.0, Qvel_scale=1.0, Qmot_scale=0.01, R_scale=1e-3,
    Qf_pos_scale=200.0, Qf_quat_scale=50.0,
    umin=None, umax=None
):
    n = x0.size
    N = U_init.shape[0]
    m = U_init.shape[1]

    # cost weights
    Qpos  = Qpos_scale  * np.eye(3)
    Qquat = Qquat_scale * np.eye(4)
    Qvel  = Qvel_scale  * np.eye(3)
    Qmot  = Qmot_scale  * np.eye(4)
    R     = R_scale     * np.eye(m)
    Qf_pos  = Qf_pos_scale  * np.eye(3)
    Qf_quat = Qf_quat_scale * np.eye(4)

    # hover quaternion and hover motor speed stored in hover[-1]

    #hover = np.array([1.,0.,0.,0., w_hover])
    q_hover = np.array([1., 0., 0., 0.])  # unit quaternion
    w_hover = env.w_hover  # scalar motor hover speed

    X = np.zeros((N+1, n))
    U = U_init.copy()

    reg = reg_init
    alphas = 0.5 ** np.arange(0, 8)  # line search

    def rollout(x0, U):
        X = np.zeros((N+1, n)); X[0] = x0
        cost = 0.0
        for k in range(N):
            u = U[k]
            if umin is not None: u = clamp(u, umin, umax)
            x_next = f(X[k], u, dt, params)
            c = running_cost(X[k], u, target_pos, q_hover, w_hover,Qpos, Qquat, Qvel, Qmot, R)
            X[k+1] = x_next; cost += c
        cost += terminal_cost(X[-1], target_pos, q_hover, Qf_pos, Qf_quat)
        return X, cost

    # initial rollout
    X, J = rollout(x0, U)

    for it in range(max_iter):
        # backward pass
        Vx, Vxx = terminal_derivatives(X[-1], target_pos, q_hover, Qf_pos, Qf_quat)
        K = np.zeros((N, m, n))
        kff = np.zeros((N, m))
        diverged = False

        for k in reversed(range(N)):
            xk = X[k]; uk = U[k]
            Ak, Bk = finite_diff_jacobian(f, xk, uk, dt, params)
            lx, lu, Lxx, Luu, Lux = cost_derivatives(xk, uk, target_pos, q_hover, w_hover, Qpos, Qquat, Qvel, Qmot, R)

            Qx  = lx  + Ak.T @ Vx
            Qu  = lu  + Bk.T @ Vx
            Qxx = Lxx + Ak.T @ Vxx @ Ak
            Quu = Luu + Bk.T @ Vxx @ Bk
            Qux = Lux + Bk.T @ Vxx @ Ak

            # regularize
            Quu_reg = Quu + reg * np.eye(m)

            # solve for gains
            try:
                # Cholesky could be used; here robust solve
                Kk = -np.linalg.solve(Quu_reg, Qux)
                kk = -np.linalg.solve(Quu_reg, Qu)
            except np.linalg.LinAlgError:
                diverged = True
                break

            # update value function
            Vx  = Qx + Kk.T @ Qu + Qux.T @ kk + Kk.T @ Quu @ kk
            Vxx = Qxx + Kk.T @ Qux + Qux.T @ Kk + Kk.T @ Quu @ Kk
            # symmetrize Vxx for numerical stability
            Vxx = 0.5 * (Vxx + Vxx.T)

            K[k] = Kk
            kff[k] = kk

        if diverged:
            reg *= 10.0
            continue

        # forward line search
        accepted = False
        for alpha in alphas:
            U_new = np.zeros_like(U)
            X_new = np.zeros_like(X)
            X_new[0] = x0
            cost_new = 0.0
            for k in range(N):
                du = alpha * kff[k] + K[k] @ (X_new[k] - X[k])
                u = U[k] + du
                if umin is not None: u = clamp(u, umin, umax)
                U_new[k] = u
                X_new[k+1] = f(X_new[k], u, dt, params)
                cost_new += running_cost(X_new[k], u, target_pos, q_hover, w_hover, Qpos, Qquat, Qvel, Qmot, R)
            cost_new += terminal_cost(X_new[-1], target_pos, q_hover, Qf_pos, Qf_quat)

            if cost_new < J:
                U = U_new; X = X_new; J = cost_new
                reg = max(reg * 0.7, 1e-9)
                accepted = True
                break

        if not accepted:
            reg *= 10.0

        # convergence check
        if accepted and np.abs(cost_new - J) < tol:
            break

    return X, U, J

if __name__ == "__main__":
    env = QuadcopterEnv(dt=0.02)
    dt = env.dt

    # --- Parameters for dynamics ---
    params = dict(
        g=env.g, mB=env.mB, dym=env.dym, dxm=env.dxm, IB=env.IB, IRzz=env.IRzz,
        Cd=env.Cd, kTh=env.kTh, kTo=env.kTo, usePrecession=env.usePrecession,
        minWmotor=env.minWmotor, maxWmotor=env.maxWmotor
    )

    # --- Initial and target states ---
    x0 = env.reset()
    x0[3:7] = np.array([1, 0, 0, 0])    # hover quaternion
    x0[0:3] = np.array([0., 0., 0.])    # start position
    x0[13:17] = np.array([env.w_hover]*4)

    N = 200
    U_init = np.zeros((N, env.action_size))
    umin = env.action_min.ravel()
    umax = env.action_max.ravel()

    target_pos = env.target_position
    w_hover = env.w_hover

    # --- Run iLQR optimization ---
    X, U, J = ilqr(
        f=f_discrete, x0=x0, U_init=U_init, dt=dt, params=params,
        target_pos=target_pos, w_hover=w_hover,
        max_iter=200, tol=1e-4,
        Qpos_scale=80.0, Qquat_scale=10.0, Qvel_scale=1.0, Qmot_scale=1e-3,
        R_scale=1e-4, Qf_pos_scale=400.0, Qf_quat_scale=50.0,
        umin=umin, umax=umax
    )


    print(f"\n✅ Final cost: {J:.3f}")
    print("Final position:", X[-1, 0:3])

    # ======================================================
    # 1️⃣ Plot 3D position trajectory
    # ======================================================
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[:, 0], X[:, 1], X[:, 2], 'b-', lw=2, label='iLQR trajectory')
    ax.scatter(X[0, 0], X[0, 1], X[0, 2], c='g', marker='o', s=80, label='Start')
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], c='r', marker='*', s=120, label='Goal')
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Quadcopter iLQR Trajectory")
    ax.legend()
    ax.grid(True)

    # ======================================================
    # 2️⃣ Plot control inputs (motor torques)
    # ======================================================
    t = np.arange(U.shape[0]) * dt
    fig2, axs = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
    for i in range(4):
        axs[i].plot(t, U[:, i], lw=1.5)
        axs[i].set_ylabel(fr"$u_{i+1}$ [Nm]")
        axs[i].grid(alpha=0.3)
    axs[-1].set_xlabel("Time [s]")
    fig2.suptitle("Optimized Motor Torques (iLQR)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # ======================================================
    # 3️⃣ Plot position and velocities
    # ======================================================
    fig3, axs3 = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axs3[0].plot(t, X[:-1, 0], label='x')
    axs3[0].plot(t, X[:-1, 1], label='y')
    axs3[0].plot(t, X[:-1, 2], label='z')
    axs3[0].set_ylabel("Position [m]")
    axs3[0].legend()
    axs3[0].grid(alpha=0.3)

    axs3[1].plot(t, X[:-1, 7], label='vx')
    axs3[1].plot(t, X[:-1, 8], label='vy')
    axs3[1].plot(t, X[:-1, 9], label='vz')
    axs3[1].set_ylabel("Velocity [m/s]")
    axs3[1].set_xlabel("Time [s]")
    axs3[1].legend()
    axs3[1].grid(alpha=0.3)
    fig3.suptitle("Quadcopter Translational States")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.show()