import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ===========================================================
# --- Scenario setup (two robots exchanging positions)
# ===========================================================
dt = 0.05
T = 150

# Initial positions and goals
x1_0, x2_0 = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
g1, g2     = np.array([2.0, 0.0]), np.array([-2.0, 0.0])

# Control limits
umin, umax = -1.5, 1.5

# Safety distance and alpha gain
d_min = 1.0
k_alpha = 2.0

# ===========================================================
# --- Helper functions
# ===========================================================
def cbf_halfspace_pair(x1, x2, u2):
    """Return halfspace (a,b) for robot 1’s CBF: a^T u1 >= b."""
    g = 2 * (x1 - x2)
    h = np.dot(x1 - x2, x1 - x2) - d_min**2
    b = -np.dot(g, u2) - k_alpha * h
    return g, b  # feasible if g^T u1 >= b

def project_to_polytope(u_des, hs):
    """Project u_des to satisfy all linear constraints aᵢᵀu ≥ bᵢ."""
    u = u_des.copy()
    for _ in range(50):
        u = 0.7*u + 0.3*u_des
        for a, b in hs:
            if a @ u < b:
                u = u + ((b - a @ u)/(a @ a))*a
        u = np.clip(u, umin, umax)
    return u

def feasible_hull(hs):
    """Return convex hull of feasible set for plotting (2D)."""
    cand = [np.array([umin, umin]), np.array([umin, umax]),
            np.array([umax, umin]), np.array([umax, umax])]
    for i in range(len(hs)):
        a1,b1 = hs[i]
        for j in range(i+1,len(hs)):
            a2,b2 = hs[j]
            A = np.stack([a1,a2])
            if abs(np.linalg.det(A))<1e-10: continue
            u = np.linalg.solve(A, np.array([b1,b2]))
            cand.append(u)
    feas = [u for u in cand if all(a@u >= b-1e-8 for a,b in hs)]
    if not feas: return np.empty((0,2))
    P = np.unique(np.round(feas,10),axis=0)
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    P = P[np.argsort(P[:,0]+1e-6*P[:,1])]
    lower,upper=[],[]
    for p in P:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0: lower.pop()
        lower.append(tuple(p))
    for p in reversed(P):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0: upper.pop()
        upper.append(tuple(p))
    return np.array(lower[:-1]+upper[:-1])

# ===========================================================
# --- Simulation loop
# ===========================================================
x1, x2 = [x1_0], [x2_0]
u1s, u2s, hulls = [], [], []

for _ in range(T):
    x1_t, x2_t = x1[-1], x2[-1]

    # Nominal desired velocities toward goals
    u1_des = (g1 - x1_t)
    u2_des = (g2 - x2_t)
    if np.linalg.norm(u1_des)>1e-6: u1_des = u1_des/np.linalg.norm(u1_des)*1.2
    if np.linalg.norm(u2_des)>1e-6: u2_des = u2_des/np.linalg.norm(u2_des)*1.2

    # Pairwise CBF constraints for each robot
    g12,b12 = cbf_halfspace_pair(x1_t, x2_t, u2_des)
    g21,b21 = -g12, -np.dot(g12, u1_des) - k_alpha*(np.dot(x2_t - x1_t, x2_t - x1_t) - d_min**2)

    # Each robot's halfspaces (CBF + box)
    hs1 = [(g12,b12),
           (np.array([1.,0.]),umin),(-np.array([1.,0.]),-umax),
           (np.array([0.,1.]),umin),(-np.array([0.,1.]),-umax)]
    hs2 = [(g21,b21),
           (np.array([1.,0.]),umin),(-np.array([1.,0.]),-umax),
           (np.array([0.,1.]),umin),(-np.array([0.,1.]),-umax)]

    # Feasible projected controls
    u1 = project_to_polytope(u1_des, hs1)
    u2 = project_to_polytope(u2_des, hs2)

    # Integrate dynamics
    x1_next, x2_next = x1_t + dt*u1, x2_t + dt*u2
    x1.append(x1_next); x2.append(x2_next)
    u1s.append(u1); u2s.append(u2)

    # Store feasible region for robot 1
    hulls.append(feasible_hull(hs1))

x1, x2, u1s, u2s = np.array(x1), np.array(x2), np.array(u1s), np.array(u2s)

# ===========================================================
# --- Animation setup
# ===========================================================
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(6,8))
fig.tight_layout(pad=3)

# Workspace plot (top)
ax1.set_xlim(-3,3); ax1.set_ylim(-2,2)
ax1.set_aspect('equal','box')
ax1.set_title("Robot trajectories and safety zone")
r1_dot, = ax1.plot([], [], 'ro', label="Robot 1")
r2_dot, = ax1.plot([], [], 'bo', label="Robot 2")
r1_path, = ax1.plot([], [], 'r-', linewidth=1)
r2_path, = ax1.plot([], [], 'b-', linewidth=1)
circle = plt.Circle((0,0), d_min/2, color='gray', fill=False, linestyle='--')
ax1.add_patch(circle)
ax1.legend()

# Control-space plot (bottom)
ax2.set_xlim(umin-0.5, umax+0.5)
ax2.set_ylim(umin-0.5, umax+0.5)
ax2.set_aspect('equal','box')
ax2.set_title("Feasible control space of Robot 1")
box_x = [umin, umax, umax, umin, umin]
box_y = [umin, umin, umax, umax, umin]
ax2.plot(box_x, box_y, 'k--', linewidth=1)
poly_fill = ax2.fill([], [], alpha=0.3, color='green')[0]
ctrl_pt, = ax2.plot([], [], 'ro', markersize=4)
time_text = ax2.text(0.05, 0.9, "", transform=ax2.transAxes)

# ===========================================================
# --- Animation update
# ===========================================================
def update(frame):
    # Update workspace
    # r1_dot.set_data(x1[frame,0], x1[frame,1])
    # r2_dot.set_data(x2[frame,0], x2[frame,1])
    # r1_path.set_data(x1[:frame,0], x1[:frame,1])
    # r2_path.set_data(x2[:frame,0], x2[:frame,1])
    # circle.center = ((x1[frame,0]+x2[frame,0])/2,
    #                  (x1[frame,1]+x2[frame,1])/2)
    circle.set_radius(d_min/2)

    # Update control-space feasible set
    hull = hulls[frame]
    if hull.shape[0]>=3:
        poly_fill.set_xy(np.vstack([hull, hull[0]]))
    else:
        poly_fill.set_xy(np.empty((0,2)))
    ctrl_pt.set_data([u1s[frame,0]], [u1s[frame,1]])
    time_text.set_text(f"t = {frame*dt:.2f}s")
    return r1_dot, r2_dot, r1_path, r2_path, poly_fill, ctrl_pt, circle, time_text

ani = FuncAnimation(fig, update, frames=T, interval=80, blit=True)
plt.show()



