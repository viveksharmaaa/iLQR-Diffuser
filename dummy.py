# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle, Polygon
# from matplotlib import animation
# import cvxpy as cp
#
# # --------- Problem Setup ---------
# dt = 0.05
# T = 200
#
# x = np.array([-4.0, -4.0])   # Start
# x_goal = np.array([4.0, 4.0])  # Goal
#
# # Obstacles: (center_x, center_y, radius)
# obstacles = [
#     (0.0, 0.0, 1.2),
#     (-2.0, 2.0, 1.0),
#     (2.0, -2.0, 1.0)
# ]
#
# alpha = 1.0  # CBF gain
#
# # Control bounds
# u_min = np.array([-1.0, -1.0])
# u_max = np.array([1.0, 1.0])
#
# def cbf_constraints(x):
#     """Return A, b for CBF linear constraints A u >= b."""
#     A = []
#     b = []
#     for (cx, cy, R) in obstacles:
#         p = x
#         p_obs = np.array([cx, cy])
#         h = np.linalg.norm(p - p_obs)**2 - R**2
#         a = 2*(p - p_obs)                 # ∇h·u
#         rhs = -alpha * h
#         A.append(a)
#         b.append(rhs)
#     return np.array(A), np.array(b)
#
# def qp_control(x):
#     """Solve QP: minimize ||u - u_des||^2 subject to CBF + bounds."""
#     u = cp.Variable(2)
#     u_des = 1.0*(x_goal - x)  # nominal controller
#
#     A, b = cbf_constraints(x)
#     constraints = [A @ u >= b,
#                    u >= u_min,
#                    u <= u_max]
#     prob = cp.Problem(cp.Minimize(cp.sum_squares(u - u_des)), constraints)
#     prob.solve(solver=cp.OSQP, warm_start=True)
#     return u.value
#
# def polygon_clip(poly, a, b):
#     """Clip polygon with halfspace a^T u >= b."""
#     clipped = []
#     for i in range(len(poly)):
#         p1 = poly[i]
#         p2 = poly[(i+1) % len(poly)]
#         v1 = a @ p1 - b
#         v2 = a @ p2 - b
#
#         if v1 >= 0:  # p1 inside
#             clipped.append(p1)
#
#         if v1 * v2 < 0:  # edge intersects boundary
#             t = v1 / (v1 - v2)
#             p_int = p1 + t*(p2 - p1)
#             clipped.append(p_int)
#
#     return np.array(clipped) if len(clipped)>0 else np.array([])
#
# def feasible_polygon(x):
#     """Compute feasible polygon of intersection."""
#     # start with control box polygon (CCW)
#     poly = np.array([
#         [u_min[0], u_min[1]],
#         [u_min[0], u_max[1]],
#         [u_max[0], u_max[1]],
#         [u_max[0], u_min[1]]
#     ])
#
#     A, b = cbf_constraints(x)
#     for ai, bi in zip(A, b):
#         poly = polygon_clip(poly, ai, bi)
#         if poly.shape[0] == 0:
#             break
#     return poly
#
# # -------- Animation Setup --------
# fig, (ax_w, ax_u) = plt.subplots(1, 2, figsize=(12,6))
#
# traj = []
#
# def update(frame):
#     global x
#
#     ax_w.clear()
#     ax_u.clear()
#
#     # Workspace plot
#     ax_w.set_xlim(-5,5)
#     ax_w.set_ylim(-5,5)
#     ax_w.set_title("Workspace")
#     ax_w.scatter(x_goal[0], x_goal[1], c='red', s=100, label="Goal")
#     ax_w.scatter(x[0], x[1], c='blue', s=100, label="Robot")
#     for (cx,cy,R) in obstacles:
#         ax_w.add_patch(Circle((cx,cy), R, fill=False, linewidth=2))
#
#     # Compute control
#     u = qp_control(x)
#     x = x + dt*u
#     traj.append(x.copy())
#     P = np.array(traj)
#     ax_w.plot(P[:,0], P[:,1], 'b--')
#
#     # Control space plot
#     ax_u.set_xlim(u_min[0]-0.2, u_max[0]+0.2)
#     ax_u.set_ylim(u_min[1]-0.2, u_max[1]+0.2)
#     ax_u.set_title("Control Space Feasible Region")
#
#     # Draw control bounds (dotted thick box)
#     bounds = np.array([
#         [u_min[0], u_min[1]],
#         [u_min[0], u_max[1]],
#         [u_max[0], u_max[1]],
#         [u_max[0], u_min[1]],
#         [u_min[0], u_min[1]]
#     ])
#     ax_u.plot(bounds[:,0], bounds[:,1], 'k:', linewidth=3)
#
#     # Feasible polygon
#     poly = feasible_polygon(x)
#     if poly.shape[0] >= 3:
#         patch = Polygon(poly, closed=True, facecolor='green', alpha=0.5)
#         ax_u.add_patch(patch)
#
#     # Current control
#     ax_u.scatter(u[0], u[1], c='red', s=70)
#
#     return []
#
# ani = animation.FuncAnimation(fig, update, frames=T, interval=60)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib import animation
import cvxpy as cp

# --------- Setup ---------
dt = 0.05
T = 200

x = np.array([-4.0, -4.0])
x_goal = np.array([4.0, 4.0])

# Obstacles arranged matching the figure (centered spread)
obstacles = [
    (-1.5, 1.2, 1.1),  # upper-left
    (0.0, -0.5, 1.1),  # center-lower
    (1.5, 1.2, 1.1)  # upper-right
]

# Colors for obstacles & CBF halfspaces (match per constraint)
obs_colors = ["red", "blue", "orange"]

alpha = 1.0
u_min = np.array([-2.5, -2.5])
u_max = np.array([2.5, 2.5])


# ----------- CBF Constraints -----------
def cbf_constraints(x):
    A, b = [], []
    for (cx, cy, R) in obstacles:
        p = x
        c = np.array([cx, cy])
        h = np.linalg.norm(p - c) ** 2 - R ** 2
        grad_h = 2 * (p - c)
        A.append(grad_h)
        b.append(-alpha * h)
    return np.array(A), np.array(b)


def qp_control(x):
    u = cp.Variable(2)
    u_des = 1.2 * (x_goal - x)
    A, b = cbf_constraints(x)
    cons = [A @ u >= b, u >= u_min, u <= u_max]
    cp.Problem(cp.Minimize(cp.sum_squares(u - u_des)), cons).solve(solver=cp.OSQP)
    return np.array(u.value)


# ------- Geometry for Feasible Set -------
def polygon_clip(poly, a, b):
    clipped = []
    for i in range(len(poly)):
        p1 = poly[i];
        p2 = poly[(i + 1) % len(poly)]
        v1 = a @ p1 - b;
        v2 = a @ p2 - b
        if v1 >= 0: clipped.append(p1)
        if v1 * v2 < 0:
            t = v1 / (v1 - v2)
            clipped.append(p1 + t * (p2 - p1))
    return np.array(clipped)


def feasible_polygon(x):
    poly = np.array([
        [u_min[0], u_min[1]],
        [u_min[0], u_max[1]],
        [u_max[0], u_max[1]],
        [u_max[0], u_min[1]]
    ])
    A, b = cbf_constraints(x)
    for ai, bi in zip(A, b):
        poly = polygon_clip(poly, ai, bi)
        if poly.shape[0] < 3:
            return np.array([])
    return poly


# Compute line segment for half-space boundary within control limits
def halfspace_line(a, b):
    pts = []
    # intersect with each box boundary
    candidates = [
        np.array([u_min[0], (b - a[0] * u_min[0]) / a[1]]),
        np.array([u_max[0], (b - a[0] * u_max[0]) / a[1]]),
        np.array([(b - a[1] * u_min[1]) / a[0], u_min[1]]),
        np.array([(b - a[1] * u_max[1]) / a[0], u_max[1]])
    ]
    for p in candidates:
        if u_min[0] - 1e-6 <= p[0] <= u_max[0] + 1e-6 and u_min[1] - 1e-6 <= p[1] <= u_max[1] + 1e-6:
            pts.append(p)
    if len(pts) >= 2:
        return np.array(pts[:2])
    return None


# --------- Animation ---------
fig, (ax_w, ax_u) = plt.subplots(1, 2, figsize=(12, 6))
traj = []


def update(frame):
    global x, traj

    # Workspace
    ax_w.clear()
    ax_w.set_xlim(-5, 5);
    ax_w.set_ylim(-5, 5);
    ax_w.set_aspect('equal')
    ax_w.set_title("Workspace with Colored Obstacles")
    ax_w.scatter(x_goal[0], x_goal[1], c='black', s=90, label="goal")

    for (cx, cy, R), col in zip(obstacles, obs_colors):
        ax_w.add_patch(Circle((cx, cy), R, color=col, alpha=0.8))

    if len(traj) > 2:
        tr = np.array(traj)
        ax_w.plot(tr[:, 0], tr[:, 1], 'g', linewidth=3)
        ax_w.scatter(tr[-1, 0], tr[-1, 1], c='red', s=70)

    # QP step
    u = qp_control(x)
    x[:] = x + dt * u
    traj.append(x.copy())

    # Control space
    ax_u.clear()
    ax_u.set_xlim(-3, 3);
    ax_u.set_ylim(-3, 3)
    ax_u.set_title("Feasible Control Region (Colored CBF Half-Spaces)")
    ax_u.set_aspect('equal')
    ax_u.grid(True, linestyle=':', linewidth=1)

    # Draw bound box (thick dashed)
    box = np.array([
        [u_min[0], u_min[1]], [u_min[0], u_max[1]], [u_max[0], u_max[1]],
        [u_max[0], u_min[1]], [u_min[0], u_min[1]]
    ])
    ax_u.plot(box[:, 0], box[:, 1], 'k--', linewidth=3)

    # Plot each constraint halfspace with same color as obstacle
    A, b = cbf_constraints(x)
    for (ai, bi), col in zip(zip(A, b), obs_colors):
        line = halfspace_line(ai, bi)
        if line is not None:
            ax_u.plot(line[:, 0], line[:, 1], color=col, linewidth=2)
            # Label line
            mx, my = np.mean(line, axis=0)
            #ax_u.text(mx, my, f"{col} obs", color=col, fontsize=9, weight='bold')

    # Feasible polygon
    poly = feasible_polygon(x)
    if poly.shape[0] >= 3:
        ax_u.add_patch(Polygon(poly, closed=True, facecolor='green', alpha=0.65, edgecolor='black'))

    # Chosen control
    ax_u.scatter(u[0], u[1], c='red', s=80)


ani = animation.FuncAnimation(fig, update, frames=T, interval=60)
plt.show()

#########################
#Moving colored obstacles

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib import animation
import cvxpy as cp

# --------- Setup ---------
dt = 0.05
T = 300

x = np.array([-4.0, -4.0])
x_goal = np.array([4.0, 4.0])

# Colors for obstacles & their half-space lines
obs_colors = ["red", "blue", "orange"]

# Safety margin (inflate obstacles by rho)
rho = 0.30
alpha = 1.1

# Control bounds
u_min = np.array([-2.5, -2.5])
u_max = np.array([2.5, 2.5])


# ---------- Moving Obstacles ----------
def obstacle_positions(t):
    """ Return list of (cx, cy, R) at time t """
    R = 1.0  # all same radius before inflation

    cx1 = -1.8 + 0.8*np.sin(0.5*t)    # red: left oscillates horizontally
    cy1 = 1.2

    cx2 = 0.0
    cy2 = -0.7 + 0.6*np.sin(0.4*t)    # blue: vertical oscillation

    cx3 = 1.5 + 0.7*np.cos(0.3*t)     # orange: circular motion
    cy3 = 1.2 + 0.7*np.sin(0.3*t)

    return [
        (cx1, cy1, R + rho),
        (cx2, cy2, R + rho),
        (cx3, cy3, R + rho)
    ]


# ---------- CBF Constraints ----------
def cbf_constraints(x, t):
    A, b = [], []
    obs = obstacle_positions(t)
    for (cx, cy, R_eff) in obs:
        p = x
        c = np.array([cx, cy])
        h = np.linalg.norm(p - c)**2 - R_eff**2
        grad_h = 2*(p - c)
        A.append(grad_h)
        b.append(-alpha*h)
    return np.array(A), np.array(b)


# ---------- QP Control (with no slack, but we can add later) ----------
def qp_control(x, t):
    u = cp.Variable(2)
    u_des = 1.2*(x_goal - x)
    A, b = cbf_constraints(x, t)
    constraints = [A@u >= b, u >= u_min, u <= u_max]
    cp.Problem(cp.Minimize(cp.sum_squares(u-u_des)), constraints).solve(solver=cp.OSQP)
    return np.array(u.value)


# ---------- Feasible Polygon ----------
def polygon_clip(poly, a, b):
    clipped=[]
    for i in range(len(poly)):
        p1 = poly[i]
        p2 = poly[(i+1)%len(poly)]
        v1 = a@p1 - b
        v2 = a@p2 - b
        if v1 >= 0:
            clipped.append(p1)
        if v1*v2 < 0:
            t = v1/(v1-v2)
            clipped.append(p1+t*(p2-p1))
    return np.array(clipped)

def feasible_polygon(x, t):
    poly = np.array([
        [u_min[0], u_min[1]],
        [u_min[0], u_max[1]],
        [u_max[0], u_max[1]],
        [u_max[0], u_min[1]]
    ])
    A,b = cbf_constraints(x,t)
    for ai,bi in zip(A,b):
        poly = polygon_clip(poly, ai, bi)
        if poly.shape[0] < 3:
            return np.array([])
    return poly


# ---------- Half-space boundary segment ----------
def halfspace_line(a, b):
    pts=[]
    candidates=[
        np.array([u_min[0], (b - a[0]*u_min[0])/a[1]]),
        np.array([u_max[0], (b - a[0]*u_max[0])/a[1]]),
        np.array([(b - a[1]*u_min[1])/a[0], u_min[1]]),
        np.array([(b - a[1]*u_max[1])/a[0], u_max[1]])
    ]
    for p in candidates:
        if u_min[0]-1e-6 <= p[0] <= u_max[0]+1e-6 and u_min[1]-1e-6 <= p[1] <= u_max[1]+1e-6:
            pts.append(p)
    if len(pts)>=2:
        return np.array(pts[:2])
    return None


# ---------- Animation ----------
fig,(ax_w,ax_u)=plt.subplots(1,2,figsize=(12,6))
traj=[]

def update(frame):
    global x, traj
    t = frame*dt

    ax_w.clear()
    ax_w.set_xlim(-5,5); ax_w.set_ylim(-5,5); ax_w.set_aspect('equal')
    ax_w.set_title("Workspace with Moving Colored Obstacles")

    # Draw moving obstacles
    obs = obstacle_positions(t)
    for (cx,cy,R_eff),col in zip(obs,obs_colors):
        ax_w.add_patch(Circle((cx,cy), R_eff, color=col, alpha=0.8))
        ax_w.add_patch(Circle((cx,cy), R_eff-rho, fill=False, linestyle='--', edgecolor=col, linewidth=2))

    # Draw path
    if len(traj)>2:
        tr=np.array(traj)
        ax_w.plot(tr[:,0], tr[:,1], 'g', linewidth=3)

    ax_w.scatter(x_goal[0], x_goal[1], c='black', s=90)
    ax_w.scatter(x[0], x[1], c='red', s=70)

    # QP step
    u = qp_control(x,t)
    x[:] = x + dt*u
    traj.append(x.copy())

    # Control space plot
    ax_u.clear()
    ax_u.set_xlim(-3,3); ax_u.set_ylim(-3,3)
    ax_u.set_aspect('equal')
    ax_u.set_title("Feasible Control Region")
    ax_u.grid(True, linestyle=':', linewidth=1)

    # Control bounds
    box = np.array([[u_min[0],u_min[1]],[u_min[0],u_max[1]],
                    [u_max[0],u_max[1]],[u_max[0],u_min[1]],[u_min[0],u_min[1]]])
    ax_u.plot(box[:,0], box[:,1], 'k--', linewidth=3)

    # Half-space boundaries
    A,b = cbf_constraints(x,t)
    for (ai,bi),col in zip(zip(A,b),obs_colors):
        line = halfspace_line(ai,bi)
        if line is not None:
            ax_u.plot(line[:,0],line[:,1],color=col,linewidth=2)

    poly = feasible_polygon(x,t)
    if poly.shape[0]>=3:
        ax_u.add_patch(Polygon(poly, closed=True, facecolor='green', alpha=0.65, edgecolor='black'))

    ax_u.scatter(u[0],u[1],c='red',s=80)


ani = animation.FuncAnimation(fig, update, frames=T, interval=60)
plt.show()
###########
#Feasible Area Code

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib import animation
import cvxpy as cp
import matplotlib.cm as cm

# =========================================
# PARAMETERS
# =========================================
dt = 0.05
T = 300
alpha = 1.1                # CBF gain
rho = 0.30                 # safety margin (inflate obstacles)
u_min = np.array([-2.5, -2.5])
u_max = np.array([2.5, 2.5])
x = np.array([-4.0, -4.0])
x_goal = np.array([4.0, 4.0])

# Colors for obstacles & their half-space boundaries
obs_colors = ["red", "blue", "orange"]

# =========================================
# MOVING OBSTACLES
# =========================================
def obstacle_positions(t):
    R = 1.0  # base radius (un-inflated)
    cx1 = -1.8 + 0.8*np.sin(0.5*t)
    cy1 = 1.2
    cx2 = 0.0
    cy2 = -0.7 + 0.6*np.sin(0.4*t)
    cx3 = 1.5 + 0.7*np.cos(0.3*t)
    cy3 = 1.2 + 0.7*np.sin(0.3*t)
    return [
        (cx1, cy1, R + rho),
        (cx2, cy2, R + rho),
        (cx3, cy3, R + rho)
    ]

# =========================================
# CBF CONSTRAINTS
# =========================================
def cbf_constraints(x, t):
    A, b = [], []
    obs = obstacle_positions(t)
    for (cx, cy, R_eff) in obs:
        c = np.array([cx, cy])
        h = np.linalg.norm(x - c)**2 - R_eff**2
        grad_h = 2*(x - c)
        A.append(grad_h)
        b.append(-alpha*h)
    return np.array(A), np.array(b)

# =========================================
# QP CONTROL (NO SLACK FOR DEMO)
# =========================================
def qp_control(x, t):
    u = cp.Variable(2)
    u_des = 1.2*(x_goal - x)
    A, b = cbf_constraints(x, t)
    constraints = [A @ u >= b, u >= u_min, u <= u_max]
    cp.Problem(cp.Minimize(cp.sum_squares(u - u_des)), constraints).solve(solver=cp.OSQP)
    return np.array(u.value)

# =========================================
# POLYGON CLIPPING (FEASIBLE SET)
# =========================================
def polygon_clip(poly, a, b):
    clipped=[]
    for i in range(len(poly)):
        p1 = poly[i]
        p2 = poly[(i+1)%len(poly)]
        v1 = a@p1 - b
        v2 = a@p2 - b
        if v1 >= 0:
            clipped.append(p1)
        if v1*v2 < 0:
            t = v1/(v1-v2)
            clipped.append(p1 + t*(p2-p1))
    return np.array(clipped)

def feasible_polygon(x, t):
    poly = np.array([
        [u_min[0], u_min[1]],
        [u_min[0], u_max[1]],
        [u_max[0], u_max[1]],
        [u_max[0], u_min[1]]
    ])
    A, b = cbf_constraints(x, t)
    for ai,bi in zip(A,b):
        poly = polygon_clip(poly, ai, bi)
        if poly.shape[0] < 3:
            return np.array([])
    return poly

# =========================================
# FEASIBLE AREA MEASUREMENT
# =========================================
def polygon_area(poly):
    if poly is None or poly.shape[0] < 3:
        return 0.0
    x = poly[:,0]
    y = poly[:,1]
    return 0.5*np.abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))

def area_color(area):
    area_norm = min(max(area / 1.5, 0), 1)
    return cm.get_cmap("RdYlGn")(area_norm)

# =========================================
# ANIMATION
# =========================================
fig,(ax_w,ax_u)=plt.subplots(1,2,figsize=(12,6))
traj=[]
area_history = []
time_history = []

def update(frame):
    global x, traj
    t = frame*dt

    # ----- Workspace -----
    ax_w.clear()
    ax_w.set_xlim(-5,5); ax_w.set_ylim(-5,5); ax_w.set_aspect('equal')
    ax_w.set_title("Workspace with Moving Colored Obstacles")

    obs = obstacle_positions(t)
    for (cx,cy,R_eff),col in zip(obs, obs_colors):
        ax_w.add_patch(Circle((cx,cy), R_eff, color=col, alpha=0.8))
        ax_w.add_patch(Circle((cx,cy), R_eff-rho, fill=False, linestyle='--', edgecolor=col, linewidth=2))

    if len(traj)>2:
        tr=np.array(traj)
        ax_w.plot(tr[:,0], tr[:,1], 'g', linewidth=3)
    ax_w.scatter(x_goal[0], x_goal[1], c='black', s=90)
    ax_w.scatter(x[0], x[1], c='red', s=70)

    # ----- Control -----
    u = qp_control(x, t)
    x[:] = x + dt*u
    traj.append(x.copy())

    # ----- Control Space -----
    ax_u.clear()
    ax_u.set_xlim(-3,3); ax_u.set_ylim(-3,3); ax_u.set_aspect('equal')
    ax_u.set_title("Feasible Control Region Shrinkage")
    ax_u.grid(True, linestyle=":", linewidth=1)

    # control bounds box
    box=np.array([[u_min[0],u_min[1]],[u_min[0],u_max[1]],[u_max[0],u_max[1]],[u_max[0],u_min[1]],[u_min[0],u_min[1]]])
    ax_u.plot(box[:,0],box[:,1],'k--',linewidth=3)

    poly = feasible_polygon(x, t)
    area = polygon_area(poly)
    area_history.append(area)
    time_history.append(t)

    # polygon shading
    if poly.shape[0]>=3:
        ax_u.add_patch(Polygon(poly, closed=True, facecolor=area_color(area), alpha=0.7, edgecolor='black'))
    else:
        ax_u.text(-1.5,-2.6,"⚠ No Feasible Control",color="red",fontsize=13,weight='bold')

    ax_u.scatter(u[0],u[1],c='red',s=80)
    ax_u.text(-2.7,2.4,f"Area = {area:.3f}",fontsize=12,weight='bold')

ani = animation.FuncAnimation(fig, update, frames=T, interval=60)
plt.show()

# =========================================
# PLOT AREA VS TIME
# =========================================
plt.figure(figsize=(7,4))
plt.plot(time_history, area_history, linewidth=3)
plt.xlabel("Time (s)")
plt.ylabel("Feasible Set Area")
plt.title("Shrinkage of Feasible Control Region Over Time")
plt.grid(True, linestyle=":", linewidth=1)
plt.show()
################

#HOCBF Unicycle dyanmics

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib import animation
import matplotlib.cm as cm
import cvxpy as cp

# =========================
# Parameters
# =========================
dt = 0.05
T = 280

# Unicycle state: [x, y, v, theta]
state = np.array([-4.0, -4.0, 0.0, np.deg2rad(45.0)])
goal = np.array([4.0, 4.0])

# Control bounds for u = [a, omega]
u_min = np.array([-2.0, -2.5])   # accel [m/s^2], yaw rate [rad/s]
u_max = np.array([ 2.0,  2.5])

# HOCBF gains
lam = 1.2
zeta = 0.9
alpha_hocbf = (lam**2)
beta_hocbf  = (2.0*zeta*lam)

# Safety margin (inflate obstacle radius)
rho = 0.30

# Obstacles: (cx, cy, R). Feel free to add more.
obstacles = [
    (-1.8,  1.3, 1.0),
    ( 0.0, -0.6, 1.0),
    ( 1.7,  1.2, 1.0),
    (-2.8, -1.0, 0.9),
    ( 2.8, -1.2, 0.9),
]
obs_colors = ["red", "blue", "orange", "purple", "teal"]

# Nominal tracking gains (for u_des)
k_v = 1.2     # speed tracking gain
k_pv = 0.6    # position → desired speed gain
k_theta = 2.0 # heading tracking

# =========================
# Helpers
# =========================
def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def polygon_clip(poly, a, b):
    """Clip convex polygon 'poly' by half-space a^T u >= b."""
    if poly.size == 0:
        return poly
    out = []
    for i in range(len(poly)):
        p1, p2 = poly[i], poly[(i+1) % len(poly)]
        v1, v2 = a @ p1 - b, a @ p2 - b
        if v1 >= 0: out.append(p1)
        if v1 * v2 < 0:
            t = v1 / (v1 - v2 + 1e-12)
            out.append(p1 + t*(p2 - p1))
    return np.array(out)

def feasible_polygon(A, b):
    """Intersect control box with all halfspaces A u >= b."""
    poly = np.array([
        [u_min[0], u_min[1]],
        [u_min[0], u_max[1]],
        [u_max[0], u_max[1]],
        [u_max[0], u_min[1]]
    ])
    for ai, bi in zip(A, b):
        poly = polygon_clip(poly, ai, bi)
        if poly.shape[0] < 3:
            return np.array([])
    return poly

def polygon_area(poly):
    if poly is None or poly.shape[0] < 3:
        return 0.0
    x = poly[:,0]; y = poly[:,1]
    return 0.5*np.abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))

def area_color(area, scale=1.5):
    a = min(max(area/scale, 0.0), 1.0)
    return cm.get_cmap("RdYlGn")(a)  # red->yellow->green

def halfspace_line(ai, bi):
    """Draw boundary (ai^T u = bi) clipped to the control box."""
    a1, a2 = ai
    pts = []
    # Intersections with x=u_min[0], x=u_max[0]
    if abs(a2) > 1e-9:
        y1 = (bi - a1*u_min[0])/a2
        y2 = (bi - a1*u_max[0])/a2
        if u_min[1]-1e-6 <= y1 <= u_max[1]+1e-6: pts.append([u_min[0], y1])
        if u_min[1]-1e-6 <= y2 <= u_max[1]+1e-6: pts.append([u_max[0], y2])
    # Intersections with y=u_min[1], y=u_max[1]
    if abs(a1) > 1e-9:
        x1 = (bi - a2*u_min[1])/a1
        x2 = (bi - a2*u_max[1])/a1
        if u_min[0]-1e-6 <= x1 <= u_max[0]+1e-6: pts.append([x1, u_min[1]])
        if u_min[0]-1e-6 <= x2 <= u_max[0]+1e-6: pts.append([x2, u_max[1]])
    if len(pts) >= 2:
        return np.array(pts[:2])
    return None

# =========================
# HOCBF (relative-degree-2) constraints for one state
# =========================
def hocbf_constraints(state):
    """
    Build A, b for constraints A @ [a, omega] >= b across all obstacles.
    HOCBF: hddot + beta*h_dot + alpha*h >= 0
    where
        h = ||r||^2 - R_eff^2
        hdot  = 2 v (r dot t)
        hddot = 2 v^2 + 2 a (r dot t) + 2 v omega (r dot n)
    """
    x, y, v, th = state
    t = np.array([np.cos(th), np.sin(th)])
    n = np.array([-np.sin(th), np.cos(th)])

    A_list, b_list = [], []
    for (cx, cy, R) in obstacles:
        R_eff = R + rho
        r = np.array([x - cx, y - cy])
        h = np.dot(r, r) - R_eff**2
        hdot = 2.0 * v * np.dot(r, t)
        # hddot = 2 v^2 + 2 a (r·t) + 2 v omega (r·n)
        c_a = 2.0 * np.dot(r, t)          # coeff for 'a'
        c_w = 2.0 * v * np.dot(r, n)      # coeff for 'omega'
        const = 2.0 * v**2

        # HOCBF: c_a a + c_w w + const + beta*hdot + alpha*h >= 0
        # -> [c_a, c_w] · u >= - const - beta*hdot - alpha*h
        A_list.append([c_a, c_w])
        b_list.append(-const - beta_hocbf*hdot - alpha_hocbf*h)

    return np.array(A_list), np.array(b_list)

# =========================
# QP Controller
# =========================
def nominal_u(state):
    """Simple goal-seeking nominal control (for cost center)."""
    x, y, v, th = state
    to_goal = goal - np.array([x, y])
    dist = np.linalg.norm(to_goal) + 1e-9
    theta_ref = np.arctan2(to_goal[1], to_goal[0])
    v_ref = np.clip(k_pv * dist, 0.0, 2.5)  # desired speed
    a_des = k_v * (v_ref - v)
    w_des = k_theta * wrap_angle(theta_ref - th)
    return np.array([a_des, w_des])

def qp_control(state):
    u = cp.Variable(2)
    u_des = nominal_u(state)
    A, b = hocbf_constraints(state)
    cons = [A @ u >= b, u >= u_min, u <= u_max]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(u - u_des)), cons)
    prob.solve(solver=cp.OSQP, warm_start=True)
    return np.array(u.value), A, b

# =========================
# Simulation + Animation
# =========================
fig, (ax_w, ax_u) = plt.subplots(1, 2, figsize=(12, 6))
traj = []
area_hist, time_hist = [], []

def update(frame):
    global state, traj
    t = frame * dt

    # --- QP step ---
    u, A, b = qp_control(state)
    if u is None or np.any(~np.isfinite(u)):
        u = np.zeros(2)

    # integrate unicycle
    x, y, v, th = state
    a, w = u
    x  += dt * v * np.cos(th)
    y  += dt * v * np.sin(th)
    v  += dt * a
    th += dt * w
    state = np.array([x, y, v, wrap_angle(th)])
    traj.append([x, y])

    # --- Workspace ---
    ax_w.clear()
    ax_w.set_xlim(-5, 5); ax_w.set_ylim(-5, 5); ax_w.set_aspect('equal')
    ax_w.set_title("Workspace (Unicycle + HOCBF-2)")

    # draw obstacles (inflated by rho)
    for (obs, col) in zip(obstacles, obs_colors):
        cx, cy, R = obs
        ax_w.add_patch(Circle((cx, cy), R + rho, color=col, alpha=0.75))
        ax_w.add_patch(Circle((cx, cy), R, fill=False, linestyle='--', edgecolor=col, linewidth=2))
    ax_w.scatter(goal[0], goal[1], c='black', s=80)
    if len(traj) > 1:
        tr = np.array(traj)
        ax_w.plot(tr[:,0], tr[:,1], 'g', linewidth=3)
        ax_w.scatter(tr[-1,0], tr[-1,1], c='red', s=70)

    # --- Control space ---
    ax_u.clear()
    ax_u.set_xlim(u_min[0]-0.1, u_max[0]+0.1)
    ax_u.set_ylim(u_min[1]-0.1, u_max[1]+0.1)
    ax_u.set_aspect('equal')
    ax_u.set_title("Feasible Control Set (a vs ω)")
    ax_u.grid(True, linestyle=":", linewidth=1)

    # bounds box (thick dashed)
    box = np.array([[u_min[0], u_min[1]],
                    [u_min[0], u_max[1]],
                    [u_max[0], u_max[1]],
                    [u_max[0], u_min[1]],
                    [u_min[0], u_min[1]]])
    ax_u.plot(box[:,0], box[:,1], 'k--', linewidth=3)

    # draw each half-space boundary in obstacle color
    for (ai, bi), col in zip(zip(A, b), obs_colors):
        line = halfspace_line(ai, bi)
        if line is not None:
            ax_u.plot(line[:,0], line[:,1], color=col, linewidth=2)

    # feasible polygon and area
    poly = feasible_polygon(A, b)
    area = polygon_area(poly)
    area_hist.append(area); time_hist.append(t)

    if poly.shape[0] >= 3:
        ax_u.add_patch(Polygon(poly, closed=True,
                               facecolor=area_color(area),
                               alpha=0.7, edgecolor='black'))
    else:
        ax_u.text(u_min[0]+0.2, u_min[1]+0.2, "⚠ No Feasible Control",
                  color="red", fontsize=12, weight='bold')

    # chosen control point
    ax_u.scatter(u[0], u[1], c='red', s=70)
    ax_u.text(u_min[0]+0.2, u_max[1]-0.3, f"Area = {area:.3f}",
              fontsize=12, weight='bold')

    return []

ani = animation.FuncAnimation(fig, update, frames=T, interval=60)
plt.show()

# =========================
# Area vs Time
# =========================
plt.figure(figsize=(7,4))
plt.plot(time_hist, area_hist, linewidth=3)
plt.xlabel("Time (s)")
plt.ylabel("Feasible Set Area")
plt.title("HOCBF-2 Feasible Control Region (Area vs Time)")
plt.grid(True, linestyle=":", linewidth=1)
plt.show()
####

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib import animation
import matplotlib.cm as cm
import cvxpy as cp

# =========================================================
# Unicycle + HOCBF (relative-degree-2) with MOVING obstacles
# =========================================================

# ---------- Parameters ----------
dt = 0.05
T = 320  # frames

# state = [x, y, v, theta]
state = np.array([-4.0, -4.0, 0.0, np.deg2rad(35.0)])
goal  = np.array([ 4.0,  4.0])

# control u = [a, omega] bounds (thick dashed box in control space)
u_min = np.array([-2.0, -2.5])
u_max = np.array([ 2.0,  2.5])

# HOCBF gains for hddot + 2*zeta*lambda hdot + lambda^2 h >= 0
lam  = 1.2
zeta = 0.95
alpha_hocbf = lam**2
beta_hocbf  = 2*zeta*lam

# Safety margin (inflate obstacles) to *encourage* infeasibility
rho = 0.35

# Colors for obstacles/half-spaces
obs_colors = ["red", "blue", "orange", "purple", "teal"]

# nominal goal-seeking gains (for cost center only)
k_v     = 1.2
k_pv    = 0.6
k_theta = 2.0

# ---------- Utilities ----------
def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def polygon_clip(poly, a, b):
    """Clip convex polygon 'poly' by half-space a^T u >= b."""
    if poly.size == 0:
        return poly
    out = []
    for i in range(len(poly)):
        p1 = poly[i]
        p2 = poly[(i+1) % len(poly)]
        v1 = a @ p1 - b
        v2 = a @ p2 - b
        if v1 >= 0: out.append(p1)
        if v1 * v2 < 0:
            t = v1 / (v1 - v2 + 1e-12)
            out.append(p1 + t*(p2 - p1))
    return np.array(out)

def feasible_polygon_from_halfspaces(A, b):
    """Intersect the control bounds box with all half-spaces A u >= b."""
    poly = np.array([
        [u_min[0], u_min[1]],
        [u_min[0], u_max[1]],
        [u_max[0], u_max[1]],
        [u_max[0], u_min[1]]
    ])
    for ai, bi in zip(A, b):
        poly = polygon_clip(poly, ai, bi)
        if poly.shape[0] < 3:
            return np.array([])
    return poly

def polygon_area(poly):
    if poly is None or poly.shape[0] < 3:
        return 0.0
    x = poly[:,0]; y = poly[:,1]
    return 0.5*np.abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))

def area_color(area, scale=1.5):
    # map area to 0..1 then red->yellow->green
    a = min(max(area/scale, 0.0), 1.0)
    return cm.get_cmap("RdYlGn")(a)

def halfspace_line(ai, bi):
    """Return segment of ai^T u = bi within the control bounds."""
    a1, a2 = ai
    pts = []
    # x = u_min[0], u_max[0]
    if abs(a2) > 1e-9:
        y1 = (bi - a1*u_min[0])/a2
        y2 = (bi - a1*u_max[0])/a2
        if u_min[1]-1e-6 <= y1 <= u_max[1]+1e-6: pts.append([u_min[0], y1])
        if u_min[1]-1e-6 <= y2 <= u_max[1]+1e-6: pts.append([u_max[0], y2])
    # y = u_min[1], u_max[1]
    if abs(a1) > 1e-9:
        x1 = (bi - a2*u_min[1])/a1
        x2 = (bi - a2*u_max[1])/a1
        if u_min[0]-1e-6 <= x1 <= u_max[0]+1e-6: pts.append([x1, u_min[1]])
        if u_min[0]-1e-6 <= x2 <= u_max[0]+1e-6: pts.append([x2, u_max[1]])
    if len(pts) >= 2:
        return np.array(pts[:2])
    return None

# ---------- Moving obstacles (designed to corner the robot) ----------
# Base radii (un-inflated); we'll draw dashed at base, solid at inflated
base_obstacles = [
    (-2.2,  1.6, 1.0),
    ( 0.0, -0.8, 1.0),
    ( 2.0,  1.4, 1.0),
    (-3.2, -1.2, 0.9),
    ( 3.0, -1.3, 0.9),
]

def obstacle_positions(t):
    """
    Return list of (cx, cy, R_base, R_eff) at time t.
    Motions are chosen so that around mid-simulation the robot is boxed in,
    making the feasible control set collapse (area -> 0).
    """
    (cx1, cy1, R1), (cx2, cy2, R2), (cx3, cy3, R3), (cx4, cy4, R4), (cx5, cy5, R5) = base_obstacles

    # Left-top sweeps horizontally toward center
    c1x = cx1 + 0.9*np.sin(0.45*t)
    c1y = cy1 + 0.15*np.sin(0.9*t)

    # Middle-bottom sweeps vertically up to pinch corridor
    c2x = cx2 + 0.25*np.sin(0.35*t + 0.8)
    c2y = cy2 + 0.9*np.sin(0.45*t + 1.0)

    # Right-top sweeps horizontally toward center
    c3x = cx3 + 0.95*np.sin(0.42*t + 0.9)
    c3y = cy3 + 0.12*np.sin(0.85*t + 0.3)

    # Far-left bottom arcs in
    c4x = cx4 + 0.8*np.cos(0.35*t + 0.3)
    c4y = cy4 + 0.5*np.sin(0.35*t + 0.3)

    # Far-right bottom arcs in
    c5x = cx5 + 0.8*np.cos(0.33*t + 1.1)
    c5y = cy5 + 0.45*np.sin(0.33*t + 1.1)

    return [
        (c1x, c1y, R1, R1+rho),
        (c2x, c2y, R2, R2+rho),
        (c3x, c3y, R3, R3+rho),
        (c4x, c4y, R4, R4+rho),
        (c5x, c5y, R5, R5+rho),
    ]

# ---------- HOCBF constraints (linear in [a, omega]) ----------
def hocbf_constraints(state, t):
    """
    Build A, b for constraints: A @ [a, omega] >= b
    From: h = ||r||^2 - R_eff^2
          hdot = 2 v (r · t_hat)
          hddot = 2 v^2 + 2 a (r · t_hat) + 2 v omega (r · n_hat)
    HOCBF: hddot + beta*hdot + alpha*h >= 0
    """
    x, y, v, th = state
    t_hat = np.array([np.cos(th), np.sin(th)])
    n_hat = np.array([-np.sin(th), np.cos(th)])

    A_list, b_list = [], []
    for (cx, cy, R_base, R_eff) in obstacle_positions(t):
        r = np.array([x - cx, y - cy])
        h = np.dot(r, r) - R_eff**2
        hdot = 2.0 * v * np.dot(r, t_hat)
        # coefficients in a, omega:
        c_a = 2.0 * np.dot(r, t_hat)
        c_w = 2.0 * v * np.dot(r, n_hat)
        const = 2.0 * v**2
        # linear inequality:
        # [c_a, c_w]·u >= -const - beta*hdot - alpha*h
        A_list.append([c_a, c_w])
        b_list.append(-const - beta_hocbf*hdot - alpha_hocbf*h)

    return np.array(A_list), np.array(b_list)

# ---------- Nominal + QP ----------
def nominal_u(state):
    """A simple goal-seeking nominal control (only for cost; safety via HOCBF)."""
    x, y, v, th = state
    to_goal = goal - np.array([x, y])
    dist = np.linalg.norm(to_goal) + 1e-9
    theta_ref = np.arctan2(to_goal[1], to_goal[0])
    v_ref = np.clip(k_pv * dist, 0.0, 2.5)
    a_des = k_v * (v_ref - v)
    w_des = k_theta * wrap_angle(theta_ref - th)
    return np.array([a_des, w_des])

def qp_control(state, t):
    # HARD constraints: no slack -> allows feasible region to go empty (area=0)
    u = cp.Variable(2)
    u_des = nominal_u(state)
    A, b = hocbf_constraints(state, t)
    constraints = [A @ u >= b, u >= u_min, u <= u_max]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(u - u_des)), constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)
    if u.value is None or not np.all(np.isfinite(u.value)):
        # no feasible solution -> return zeros so sim continues (and we can show failure)
        return np.zeros(2), A, b, False
    return np.array(u.value), A, b, True

# ---------- Animation ----------
fig, (ax_w, ax_u) = plt.subplots(1, 2, figsize=(12, 6))
traj = []
area_hist, time_hist = [], []
infeasible_frames = 0

def update(frame):
    global state, traj, infeasible_frames
    t = frame * dt

    # QP step
    u, A, B, feasible = qp_control(state, t)
    if not feasible:
        infeasible_frames += 1

    # Integrate unicycle
    x, y, v, th = state
    a, w = u
    x  += dt * v * np.cos(th)
    y  += dt * v * np.sin(th)
    v  += dt * a
    th += dt * w
    state = np.array([x, y, v, wrap_angle(th)])
    traj.append([x, y])

    # -------- Workspace --------
    ax_w.clear()
    ax_w.set_xlim(-5, 5); ax_w.set_ylim(-5, 5); ax_w.set_aspect('equal')
    ax_w.set_title("Workspace (Unicycle + HOCBF-2 with MOVING Obstacles)")

    # Draw moving obstacles: solid = inflated (R+rho); dashed = base R
    moving = obstacle_positions(t)
    for (cx, cy, R_base, R_eff), col in zip(moving, obs_colors):
        ax_w.add_patch(Circle((cx, cy), R_eff, color=col, alpha=0.75))
        ax_w.add_patch(Circle((cx, cy), R_base, fill=False, linestyle='--', edgecolor=col, linewidth=2))

    # Path & markers
    ax_w.scatter(goal[0], goal[1], c='black', s=80)
    if len(traj) > 1:
        tr = np.array(traj)
        ax_w.plot(tr[:,0], tr[:,1], 'g', linewidth=3)
    ax_w.scatter(state[0], state[1], c='red', s=70)

    # If we spent long time infeasible, announce likely failure
    if infeasible_frames * dt > 3.0:
        ax_w.text(-4.8, 4.4, "Feasibility lost → goal likely unreachable",
                  color="crimson", weight='bold', fontsize=11)

    # -------- Control Space --------
    ax_u.clear()
    ax_u.set_xlim(u_min[0]-0.1, u_max[0]+0.1)
    ax_u.set_ylim(u_min[1]-0.1, u_max[1]+0.1)
    ax_u.set_aspect('equal')
    ax_u.set_title("Control Feasible Region (a vs ω)")
    ax_u.grid(True, linestyle=":", linewidth=1)

    # bounds box (thick dashed)
    box = np.array([[u_min[0], u_min[1]],
                    [u_min[0], u_max[1]],
                    [u_max[0], u_max[1]],
                    [u_max[0], u_min[1]],
                    [u_min[0], u_min[1]]])
    ax_u.plot(box[:,0], box[:,1], 'k--', linewidth=3)

    # Half-space boundaries in obstacle colors
    for (ai, bi), col in zip(zip(A, B), obs_colors):
        line = halfspace_line(ai, bi)
        if line is not None:
            ax_u.plot(line[:,0], line[:,1], color=col, linewidth=2)

    # Feasible polygon & area (may be empty)
    poly = feasible_polygon_from_halfspaces(A, B)
    area = polygon_area(poly)
    area_hist.append(area)
    time_hist.append(t)

    if poly.shape[0] >= 3:
        ax_u.add_patch(Polygon(poly, closed=True,
                               facecolor=area_color(area),
                               alpha=0.7, edgecolor='black'))
    else:
        ax_u.text(u_min[0]+0.15, u_min[1]+0.2, "⚠ Feasible Region = 0",
                  color="red", fontsize=12, weight='bold')

    ax_u.scatter(u[0], u[1], c='red', s=70)
    ax_u.text(u_min[0]+0.2, u_max[1]-0.35, f"Area = {area:.3f}",
              fontsize=12, weight='bold')

    return []

ani = animation.FuncAnimation(fig, update, frames=T, interval=60)
plt.show()

# -------- Area vs Time --------
plt.figure(figsize=(7,4))
plt.plot(time_hist, area_hist, linewidth=3)
plt.xlabel("Time (s)")
plt.ylabel("Feasible Set Area")
plt.title("Feasible Control Region Area vs Time (Moving Obstacles, Hard HOCBF)")
plt.grid(True, linestyle=":", linewidth=1)
plt.show()

# -------- Report outcome in console --------
final_dist = np.linalg.norm(state[:2] - goal)
print(f"Final distance to goal: {final_dist:.3f} m")
if final_dist > 0.5:
    print("Result: FAILED to reach goal (as intended). Feasibility collapses to zero for prolonged periods.")
else:
    print("Result: Reached goal.")
