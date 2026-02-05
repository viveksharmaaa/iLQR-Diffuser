import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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


# # =============================
# # --- Scenario setup ---
# # =============================
# dt = 0.1
# T = 100
# x0   = np.array([-2.5, -2.0])
# goal = np.array([ 2.2,  2.3])
#
# obstacles = [
#     (np.array([-0.6,  0.4]), 0.65),
#     (np.array([ 0.6,  0.6]), 0.65),
#     (np.array([ 0.4, -0.6]), 0.65),
# ]
#
# umin, umax = -2.0, 2.0
# k_alpha = 1.0
#
# # =============================
# # --- Helper functions ---
# # =============================
# def cbf_halfspaces(x):
#     hs = []
#     for c, R in obstacles:
#         g = 2.0 * (x - c)
#         h = float((x - c) @ (x - c) - R**2)
#         hs.append((g, -k_alpha*h))
#     hs.extend([
#         (np.array([ 1., 0.]),  umin),
#         (np.array([-1., 0.]), -umax),
#         (np.array([ 0., 1.]),  umin),
#         (np.array([ 0.,-1.]), -umax),
#     ])
#     return hs
#
# def project_to_polytope(u_des, hs):
#     u = u_des.copy()
#     for _ in range(50):
#         u = 0.7*u + 0.3*u_des
#         for a, b in hs:
#             if a @ u < b:
#                 u = u + ((b - a @ u)/(a @ a))*a
#     return u
#
# def simulate():
#     xs, us = [x0.copy()], []
#     for _ in range(T):
#         x = xs[-1]
#         u_des = (goal - x)
#         if np.linalg.norm(u_des) > 1e-6:
#             u_des = u_des / np.linalg.norm(u_des) * 1.8
#         hs = cbf_halfspaces(x)
#         u = project_to_polytope(u_des, hs)
#         u = np.clip(u, umin, umax)
#         us.append(u)
#         xs.append(x + dt*u)
#     return np.array(xs), np.array(us)
#
# def feasible_hull(hs):
#     cand = [np.array([umin, umin]), np.array([umin, umax]),
#             np.array([umax, umin]), np.array([umax, umax])]
#     for i in range(len(hs)):
#         a1,b1 = hs[i]
#         for j in range(i+1, len(hs)):
#             a2,b2 = hs[j]
#             A = np.stack([a1,a2])
#             if abs(np.linalg.det(A))<1e-10: continue
#             u = np.linalg.solve(A, np.array([b1,b2]))
#             cand.append(u)
#     feas=[]
#     for u in cand:
#         if all(a@u >= b-1e-8 for a,b in hs):
#             feas.append(u)
#     if not feas: return np.empty((0,2))
#     P=np.unique(np.round(feas,10),axis=0)
#     def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
#     P=P[np.argsort(P[:,0]+1e-6*P[:,1])]
#     lower=[]
#     for p in P:
#         while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0: lower.pop()
#         lower.append(tuple(p))
#     upper=[]
#     for p in reversed(P):
#         while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0: upper.pop()
#         upper.append(tuple(p))
#     hull=np.array(lower[:-1]+upper[:-1])
#     return hull
#
# # =============================
# # --- Simulation ---
# # =============================
# xs, us = simulate()
# hulls = [feasible_hull(cbf_halfspaces(x)) for x in xs]
#
# # =============================
# # --- Animation ---
# # =============================
# fig, ax = plt.subplots(figsize=(5,5))
# ax.set_xlim(umin-1, umax+1)
# ax.set_ylim(umin-1, umax+1)
# ax.set_aspect('equal', 'box')
# ax.set_xlabel("X Velocity")
# ax.set_ylabel("Y Velocity")
# poly_fill = ax.fill([], [], alpha=0.3)[0]
# ctrl_pt, = ax.plot([], [], 'ro', markersize=5)
# text = ax.text(0.05, 0.9, "", transform=ax.transAxes)
#
# def update(frame):
#     h = hulls[frame]
#     if h.shape[0]>=3:
#         poly_fill.set_xy(np.vstack([h, h[0]]))
#     else:
#         poly_fill.set_xy(np.empty((0,2)))
#     ctrl_pt.set_data([us[frame,0]], [us[frame,1]])
#     text.set_text(f"t={frame*dt:.1f}s")
#     return poly_fill, ctrl_pt, text
#
# ani = FuncAnimation(fig, update, frames=len(xs)-1, interval=120, blit=True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =============================
# --- Scenario setup ---
# =============================
dt = 0.1
T = 100
x0   = np.array([-2.5, -2.0])
goal = np.array([ 2.2,  2.3])

# Obstacles (center, radius)
obstacles = [
    (np.array([-0.6,  0.4]), 0.65),
    (np.array([ 0.6,  0.6]), 0.65),
    (np.array([ 0.4, -0.6]), 0.65),
]

# Velocity (input) bounds
umin, umax = -2.0, 2.0
k_alpha = 1.0   # α gain in CBF

# =============================
# --- Helper functions ---
# =============================
def cbf_halfspaces(x):
    """Return halfspaces aᵢᵀ u ≥ bᵢ for all CBFs + box bounds."""
    hs = []
    for c, R in obstacles:
        g = 2.0 * (x - c)
        h = float((x - c) @ (x - c) - R**2)
        hs.append((g, -k_alpha * h))
    # Add box bounds
    hs.extend([
        (np.array([ 1., 0.]),  umin),
        (np.array([-1., 0.]), -umax),
        (np.array([ 0., 1.]),  umin),
        (np.array([ 0.,-1.]), -umax),
    ])
    return hs

def project_to_polytope(u_des, hs):
    """Project u_des into feasible set."""
    u = u_des.copy()
    for _ in range(50):
        u = 0.7*u + 0.3*u_des
        for a, b in hs:
            if a @ u < b:
                u = u + ((b - a @ u)/(a @ a))*a
    return u

def feasible_hull(hs):
    """Compute convex hull of feasible region from halfspaces (2D)."""
    cand = [np.array([umin, umin]), np.array([umin, umax]),
            np.array([umax, umin]), np.array([umax, umax])]
    for i in range(len(hs)):
        a1, b1 = hs[i]
        for j in range(i+1, len(hs)):
            a2, b2 = hs[j]
            A = np.stack([a1, a2])
            if abs(np.linalg.det(A)) < 1e-10:
                continue
            u = np.linalg.solve(A, np.array([b1, b2]))
            cand.append(u)
    feas = []
    for u in cand:
        if all(a @ u >= b - 1e-8 for a, b in hs):
            feas.append(u)
    if not feas:
        return np.empty((0,2))
    P = np.unique(np.round(np.array(feas), 10), axis=0)
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    P = P[np.argsort(P[:,0] + 1e-6*P[:,1])]
    lower, upper = [], []
    for p in P:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0: lower.pop()
        lower.append(tuple(p))
    for p in reversed(P):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0: upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1])
    return hull

# =============================
# --- Simulate agent ---
# =============================
def simulate():
    xs, us = [x0.copy()], []
    for _ in range(T):
        x = xs[-1]
        u_des = goal - x
        if np.linalg.norm(u_des) > 1e-6:
            u_des = u_des / np.linalg.norm(u_des) * 1.8
        hs = cbf_halfspaces(x)
        u = project_to_polytope(u_des, hs)
        u = np.clip(u, umin, umax)
        us.append(u)
        xs.append(x + dt*u)
    return np.array(xs), np.array(us)

xs, us = simulate()
hulls = [feasible_hull(cbf_halfspaces(x)) for x in xs]

# =============================
# --- Animation ---
# =============================
fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlim(umin-1, umax+1)
ax.set_ylim(umin-1, umax+1)
ax.set_aspect('equal', 'box')
ax.set_xlabel("X Velocity")
ax.set_ylabel("Y Velocity")

# Plot box constraint (dotted)
ax.plot([umin, umax, umax, umin, umin],
        [umin, umin, umax, umax, umin],
        'k--', linewidth=1.2, label='Input bounds')

poly_fill = ax.fill([], [], alpha=0.3, label='Feasible region')[0]
ctrl_pt, = ax.plot([], [], 'ro', markersize=5, label='Chosen control')
text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

def update(frame):
    h = hulls[frame]
    if h.shape[0]>=3:
        poly_fill.set_xy(np.vstack([h, h[0]]))
    else:
        poly_fill.set_xy(np.empty((0,2)))
    ctrl_pt.set_data([us[frame,0]], [us[frame,1]])
    text.set_text(f"t={frame*dt:.1f}s")
    return poly_fill, ctrl_pt, text

ani = FuncAnimation(fig, update, frames=len(xs)-1, interval=120, blit=True)
plt.legend()
plt.show()

