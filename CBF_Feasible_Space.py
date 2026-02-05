# Feasible-set animation for CBF-QP with a unicycle robot (controls: linear acceleration a and angular velocity ω)
# - Workspace (left): robot and moving circular obstacles
# - Control space (right): polygon = intersection of box constraints and HOCBF half-planes
#
# Notes:
#   * Uses a relative-degree-2 HOCBF on h = ||p - p_obs||^2 - R^2
#     -> \ddot h + 2*zeta*lambda * \dot h + lambda^2 * h >= 0
#     -> linear in u = [a, ω].
#   * Only numpy/matplotlib are required.
#
# Tip: In a notebook, end with "plt.show()" instead of saving to file.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib import animation
import matplotlib.pyplot as plt
# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = "palatino"
# plt.rcParams.update({
#     "font.size": 12,         # default text size
#     "axes.titlesize": 16,    # title
#     "axes.labelsize": 14,    # x and y labels
#     "xtick.labelsize": 12,   # x tick labels
#     "ytick.labelsize": 12,   # y tick labels
#     "legend.fontsize": 12,   # legend
#     "font.weight": "bold",
# })

# -------------------------- CONFIG --------------------------
np.random.seed(0)

DT = 0.05         # simulation step [s]
T  = 12.0         # total time [s]
STEPS = int(T/DT)

# Robot parameters
a_min, a_max = -4.0, 4.0
w_min, w_max = -4.0, 4.0
# rows are [A_x, A_y, b] for A u <= b
u_box = np.array([
    [ 1,  0, -a_min],  #  +a >= a_min  ->  -a <= -a_min
    [-1,  0,  a_max],  #  -a >= -a_max ->   a <= a_max
    [ 0,  1, -w_min],
    [ 0, -1,  w_max]
], dtype=float)

# CBF parameters
R_safe = 0.6          # safety radius (robot-to-obstacle)
lam    = 2.0          # HOCBF λ (>0)
zeta   = 1.0          # damping ratio (>=0.5 typical)

# Workspace params
robot_radius = 0.12

# Obstacles (moving discs): (px, py, vx, vy, radius)
OBSTS = [
    [-1.6,  2.2,  0.2, -0.15, 0.35],
    [-1.1,  1.1,  0.15,  0.00, 0.35],
    [-1.8, -0.4,  0.25,  0.12, 0.35],
    [ 1.2, -1.1, -0.15,  0.20, 0.35],
]

# World bounds for plotting & obstacle wrap-around
XMIN, XMAX = -2.5, 2.5
YMIN, YMAX = -2.5, 2.5

# Nominal (reference) control policy (simple go-to-goal)
goal = np.array([1.6, 1.2])

# ----------------------- Helper functions -------------------
def wrap(val, lo, hi):
    r = hi - lo
    return lo + ((val - lo) % r)

def sutherland_hodgman(poly, A, b):
    """
    Clip a convex polygon 'poly' (Nx2 array, CCW) by half-plane A[0:2]·u <= b.
    Returns new polygon (possibly empty).
    """
    if poly.size == 0:
        return poly
    new_pts = []
    n = len(poly)
    for i in range(n):
        P = poly[i]
        Q = poly[(i+1) % n]
        fP = A[0]*P[0] + A[1]*P[1] - b
        fQ = A[0]*Q[0] + A[1]*Q[1] - b
        insideP = fP <= 1e-9
        insideQ = fQ <= 1e-9
        if insideP and insideQ:
            new_pts.append(Q)
        elif insideP and not insideQ:
            t = fP/(fP - fQ + 1e-12)
            I = P + t*(Q - P)
            new_pts.append(I)
        elif (not insideP) and insideQ:
            t = fP/(fP - fQ + 1e-12)
            I = P + t*(Q - P)
            new_pts.append(I)
            new_pts.append(Q)
        # else: both outside
    return np.array(new_pts)

def cbf_halfspace_terms(px, py, th, v, obs_px, obs_py, obs_vx, obs_vy, R):
    """
    Build a single HOCBF inequality of form:
        A_cbf @ [a, ω] <= b_cbf
    derived from  \ddot h + 2*zeta*lam \dot h + lam^2 h >= 0,
    where h = ||r||^2 - R^2 with r = p - p_obs.
    """
    # Geometry
    r = np.array([px - obs_px, py - obs_py])
    e = np.array([np.cos(th), np.sin(th)])
    e_perp = np.array([-np.sin(th), np.cos(th)])
    p_dot = v * e
    r_dot = p_dot - np.array([obs_vx, obs_vy])

    h = np.dot(r, r) - R**2
    hdot = 2.0 * np.dot(r, r_dot)

    # r_ddot = a*e + v*ω*e_perp  (obs assumed const-vel)
    # hddot  = 2||r_dot||^2 + 2 r·(a e + v ω e_perp)
    const_terms = 2.0*np.dot(r_dot, r_dot)
    A_a = 2.0 * np.dot(r, e)             # coefficient for "a"
    A_w = 2.0 * v * np.dot(r, e_perp)    # coefficient for "ω"

    # Inequality: hddot + 2*zeta*lam*hdot + lam^2*h >= 0
    rhs = -(const_terms + 2.0*zeta*lam*hdot + (lam**2)*h)

    # Put into A u <= b by multiplying both sides by -1:
    A = np.array([-A_a, -A_w], dtype=float)
    b = -rhs
    return A, b

def feasible_polygon_from_halfspaces(Ab_list, box_rect):
    """
    Intersect a starting rectangle polygon (box_rect: 4x2 CCW) with linear halfspaces A u <= b
    given as a list of (A (2,), b).
    Returns polygon (k x 2) possibly empty.
    """
    poly = box_rect.copy()
    for A, b in Ab_list:
        poly = sutherland_hodgman(poly, A, b)
        if poly.size == 0:
            break
    return poly

# ----------------------- Simulation setup -------------------
# Initial robot state
px, py = -1.8, -1.0
th, v   = 0.2,  0.8

# Obstacles array
obsts = np.array(OBSTS, dtype=float)

# Control-space rectangle polygon
box_poly = np.array([
    [a_min, w_min],
    [a_max, w_min],
    [a_max, w_max],
    [a_min, w_max]
], dtype=float)

# -------------------------- Plot init -----------------------
fig, (ax_w, ax_u) = plt.subplots(1, 2, figsize=(10, 4.5))

# Workspace axes
ax_w.set_aspect("equal")
ax_w.set_xlim(XMIN, XMAX)
ax_w.set_ylim(YMIN, YMAX)
ax_w.set_title("Workspace")
robot_artist = Circle((px, py), radius=robot_radius, fill=False, lw=2)
ax_w.add_patch(robot_artist)
heading_line, = ax_w.plot([], [], lw=2)
goal_pt, = ax_w.plot([goal[0]], [goal[1]], marker="x", ms=8)

obs_artists = []
for (ox, oy, _, _, ro) in obsts:
    c = Circle((ox, oy), radius=ro, fill=True, alpha=0.25)
    ax_w.add_patch(c)
    obs_artists.append(c)

# Control-space axes
ax_u.set_aspect("equal")
ax_u.set_xlim(a_min-0.5, a_max+0.5)
ax_u.set_ylim(w_min-0.5, w_max+0.5)
ax_u.set_xlabel("Linear acceleration a")
ax_u.set_ylabel("Angular velocity ω")
ax_u.set_title("Feasible set in control space")

# Draw box limits as dashed rectangle
box_patch = Polygon(box_poly, closed=True, fill=False, ls="--")
ax_u.add_patch(box_patch)

feas_patch = Polygon(np.empty((0,2)), closed=True, alpha=0.35)  # feasible polygon
ax_u.add_patch(feas_patch)
nominal_scatter = ax_u.scatter([], [])
time_text = ax_u.text(0.02, 0.96, "", transform=ax_u.transAxes, va="top")

# --------------------- Dynamics & policy --------------------
def step_dynamics(px, py, th, v, a, w):
    v_new = v + DT*a
    th_new = th + DT*w
    p_new = np.array([px, py]) + DT * v * np.array([np.cos(th), np.sin(th)])
    return p_new[0], p_new[1], th_new, v_new

def nominal_policy(px, py, th, v):
    # Simple go-to-goal on position using PD on heading & speed
    to_goal = goal - np.array([px, py])
    desired_heading = np.arctan2(to_goal[1], to_goal[0])
    heading_err = np.arctan2(np.sin(desired_heading - th), np.cos(desired_heading - th))
    dist = np.linalg.norm(to_goal)
    v_des = 1.2 * np.tanh(0.8 * dist)
    a_ref = 2.0*(v_des - v)
    w_ref = 2.5*heading_err
    # Clip to bounds (nominal may be infeasible)
    return np.clip(a_ref, a_min, a_max), np.clip(w_ref, w_min, w_max)

# --------------------- Animation step -----------------------
def update(frame):
    global px, py, th, v, obsts

    # Move obstacles (wrap-around world)
    for i in range(len(obsts)):
        obsts[i,0] = wrap(obsts[i,0] + DT*obsts[i,2], XMIN, XMAX)
        obsts[i,1] = wrap(obsts[i,1] + DT*obsts[i,3], YMIN, YMAX)

    # Build halfspaces: control box + HOCBFs
    Ab_list = []
    # box rows: Ax, Ay, b
    for row in u_box:
        Ab_list.append((row[:2], row[2]))

    # CBF constraints for each obstacle
    for (ox, oy, ovx, ovy, ro) in obsts:
        A, b = cbf_halfspace_terms(px, py, th, v, ox, oy, ovx, ovy, R_safe + ro + robot_radius)
        Ab_list.append((A, b))

    # Intersect to form feasible polygon
    feas_poly = feasible_polygon_from_halfspaces(Ab_list, box_poly)

    # Nominal control
    a_ref, w_ref = nominal_policy(px, py, th, v)
    u_nom = np.array([a_ref, w_ref])
    u_apply = u_nom.copy()

    # If nominal is outside polygon, use centroid of feasible polygon as a crude projection
    if feas_poly.size == 0:
        u_apply = np.array([0.0, 0.0])  # no feasible control: stop
    else:
        # winding test
        x, y = u_nom
        P = feas_poly
        j = len(P) - 1
        c = False
        for i in range(len(P)):
            xi, yi = P[i]
            xj, yj = P[j]
            if ((yi>y) != (yj>y)) and (x < (xj-xi)*(y-yi)/(yj-yi+1e-12) + xi):
                c = not c
            j = i
        inside = c
        if not inside:
            u_apply = np.mean(P, axis=0)

    # Apply chosen control to the robot
    px, py, th, v = step_dynamics(px, py, th, v, u_apply[0], u_apply[1])

    # ---- Update artists
    robot_artist.center = (px, py)
    head = np.array([px, py]) + 0.3*np.array([np.cos(th), np.sin(th)])
    heading_line.set_data([px, head[0]], [py, head[1]])
    for k, (ox, oy, *_rest) in enumerate(obsts):
        obs_artists[k].center = (ox, oy)

    # control-space visuals
    if feas_poly.size > 0:
        feas_patch.set_xy(feas_poly)
        feas_patch.set_visible(True)
    else:
        feas_patch.set_visible(False)

    nominal_scatter.set_offsets(np.array([[u_nom[0], u_nom[1]]]))
    time_text.set_text(f"t = {frame*DT:.2f} s")
    return (*obs_artists, robot_artist, heading_line, feas_patch, nominal_scatter, time_text)

anim = animation.FuncAnimation(fig, update, frames=STEPS, interval=25, blit=False)

# In a script: save (requires ffmpeg or pillow)
from matplotlib.animation import FFMpegWriter, PillowWriter
# anim.save("cbf_feasible_set.mp4", writer=FFMpegWriter(fps=int(1/DT)))
anim.save("cbf_feasible_set.gif", writer=PillowWriter(fps=int(1/DT)))

plt.show()

