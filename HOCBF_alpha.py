import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib import animation
import matplotlib.cm as cm
import cvxpy as cp
from matplotlib.animation import PillowWriter

# =========================================================
# Unicycle + HOCBF-2 with MOVING obstacles
# =========================================================

import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# ---------- Simulation Parameters ----------
dt = 0.05
T = 280  # frames

# Compare different λ (thus α = λ² and β = 2ζλ)
lambda_values = [0.8, 2.0]   # You can add more
zeta = 0.9                   # damping

rho = 0.35                   # safety margin inflation
u_min = np.array([-2.0, -2.5])
u_max = np.array([ 2.0,  2.5])
goal  = np.array([4.0, 4.0])

# Initial state for all runs
def init_state():
    return np.array([-4.0, -4.0, 0.0, np.deg2rad(35.0)])

# Obstacles (center + radius) BEFORE motion
base_obstacles = [
    (-2.2,  1.6, 1.0),
    ( 0.0, -0.8, 1.0),
    ( 2.0,  1.4, 1.0),
    (-3.2, -1.2, 0.9),
    ( 3.0, -1.3, 0.9),
]
obs_colors = ["red","blue","orange","purple","teal"]

def ob_pos(t):
    """Return moving obstacles: (cx, cy, R_base, R_eff)."""
    (cx1, cy1, R1), (cx2, cy2, R2), (cx3, cy3, R3), (cx4, cy4, R4), (cx5, cy5, R5) = base_obstacles
    c1 = (cx1+0.9*np.sin(.45*t), cy1+0.15*np.sin(.9*t), R1, R1+rho)
    c2 = (cx2+0.25*np.sin(.35*t+.8), cy2+0.9*np.sin(.45*t+1.0), R2, R2+rho)
    c3 = (cx3+0.95*np.sin(.42*t+.9), cy3+0.12*np.sin(.85*t+.3), R3, R3+rho)
    c4 = (cx4+0.8*np.cos(.35*t+.3), cy4+0.5*np.sin(.35*t+.3), R4, R4+rho)
    c5 = (cx5+0.8*np.cos(.33*t+1.1), cy5+0.45*np.sin(.33*t+1.1), R5, R5+rho)
    return [c1,c2,c3,c4,c5]

def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi

def nominal_u(state):
    x,y,v,th = state
    d = goal - np.array([x,y])
    dist = np.linalg.norm(d)+1e-9
    th_ref = np.arctan2(d[1],d[0])
    v_ref  = np.clip(0.8*dist,0,2.5)
    a_des  = 1.2*(v_ref - v)
    w_des  = 2.0*wrap(th_ref - th)
    return np.array([a_des, w_des])

def hocbf_constraints(state, t, lam):
    alpha = lam**2
    beta  = 2*zeta*lam
    x,y,v,th = state
    t_hat = np.array([np.cos(th), np.sin(th)])
    n_hat = np.array([-np.sin(th), np.cos(th)])
    A=[]; B=[]
    for (cx,cy,Rb,Re) in ob_pos(t):
        r = np.array([x-cx, y-cy])
        h    = np.dot(r,r) - Re**2
        hdot = 2*v*np.dot(r,t_hat)
        c_a  = 2*np.dot(r,t_hat)
        c_w  = 2*v*np.dot(r,n_hat)
        const = 2*v*v
        A.append([c_a,c_w])
        B.append(-const - beta*hdot - alpha*h)
    return np.array(A), np.array(B)

def qp_control(state, t, lam):
    u = cp.Variable(2)
    u_des = nominal_u(state)
    A,B = hocbf_constraints(state,t,lam)
    cons=[A@u >= B, u>=u_min, u<=u_max]
    prob=cp.Problem(cp.Minimize(cp.sum_squares(u-u_des)),cons)
    prob.solve(solver=cp.OSQP,warm_start=True)
    if u.value is None or not np.all(np.isfinite(u.value)):
        return np.zeros(2), A, B, False
    return np.array(u.value), A, B, True

def feas_poly(A,B):
    poly=np.array([
        [u_min[0],u_min[1]],
        [u_min[0],u_max[1]],
        [u_max[0],u_max[1]],
        [u_max[0],u_min[1]]
    ])
    def clip(poly,a,b):
        out=[]
        for i in range(len(poly)):
            p1,p2=poly[i],poly[(i+1)%len(poly)]
            v1=a@p1-b; v2=a@p2-b
            if v1>=0: out.append(p1)
            if v1*v2<0:
                t=v1/(v1-v2+1e-12)
                out.append(p1+t*(p2-p1))
        return np.array(out)
    for ai,bi in zip(A,B):
        poly=clip(poly,ai,bi)
        if poly.shape[0]<3: return np.array([])
    return poly

def area(poly):
    if poly.shape[0]<3: return 0
    x=poly[:,0]; y=poly[:,1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1)) - np.dot(y,np.roll(x,1)))

def halfspace_line(ai,bi):
    a1,a2=ai
    pts=[]
    if abs(a2)>1e-9:
        y1=(bi-a1*u_min[0])/a2; y2=(bi-a1*u_max[0])/a2
        if u_min[1]<=y1<=u_max[1]: pts.append([u_min[0],y1])
        if u_min[1]<=y2<=u_max[1]: pts.append([u_max[0],y2])
    if abs(a1)>1e-9:
        x1=(bi-a2*u_min[1])/a1; x2=(bi-a2*u_max[1])/a1
        if u_min[0]<=x1<=u_max[0]: pts.append([x1,u_min[1]])
        if u_min[0]<=x2<=u_max[0]: pts.append([x2,u_max[1]])
    return np.array(pts[:2]) if len(pts)>=2 else None

# ---------- Prepare Multi-Case Animation ----------
num = len(lambda_values)
fig, axs = plt.subplots(num, 2, figsize=(12, 5*num))
#fig.tight_layout(pad=0)

states=[init_state() for _ in lambda_values]
traj=[[] for _ in lambda_values]
areas=[[] for _ in lambda_values]
times=[]

# def update(frame):
#     t=frame*dt
#     times.append(t)
#
#     for i,lam in enumerate(lambda_values):
#         ax_w, ax_u = axs[i]
#
#         # QP control
#         u,A,B,feas = qp_control(states[i],t,lam)
#         x,y,v,th = states[i]
#         x+=dt*v*np.cos(th); y+=dt*v*np.sin(th)
#         v+=dt*u[0]; th+=dt*u[1]
#         states[i]=np.array([x,y,v,wrap(th)])
#         traj[i].append([x,y])
#
#         # Workspace
#         ax_w.clear()
#         ax_w.set_xlim(-5,5); ax_w.set_ylim(-5,5); ax_w.set_aspect('equal')
#         ax_w.set_title(f"Workspace (λ={lam})")
#         for (cx,cy,Rb,Re),col in zip(ob_pos(t),obs_colors):
#             ax_w.add_patch(Circle((cx,cy),Re,color=col,alpha=0.75))
#             ax_w.add_patch(Circle((cx,cy),Rb,fill=False,linestyle='--',edgecolor=col,linewidth=2))
#         if len(traj[i])>2:
#             tr=np.array(traj[i])
#             ax_w.plot(tr[:,0],tr[:,1],'g',linewidth=3)
#         ax_w.scatter(goal[0],goal[1],c='black',s=80)
#         ax_w.scatter(states[i][0],states[i][1],c='red',s=70)
#
#         # Control space
#         ax_u.clear()
#         ax_u.set_xlim(u_min[0],u_max[0]); ax_u.set_ylim(u_min[1],u_max[1])
#         ax_u.set_aspect('equal')
#         ax_u.set_title(f"Feasible Set (λ={lam})")
#         ax_u.grid(True,linestyle=":",linewidth=1)
#
#         box=np.array([[u_min[0],u_min[1]],[u_min[0],u_max[1]],
#                       [u_max[0],u_max[1]],[u_max[0],u_min[1]],[u_min[0],u_min[1]]])
#         ax_u.plot(box[:,0],box[:,1],'k--',linewidth=3)
#
#         for (ai,bi),col in zip(zip(A,B),obs_colors):
#             ln = halfspace_line(ai,bi)
#             if ln is not None:
#                 ax_u.plot(ln[:,0],ln[:,1],color=col,linewidth=2)
#
#         poly = feas_poly(A,B)
#         a = area(poly)
#         areas[i].append(a)
#
#         if poly.shape[0]>=3:
#             ax_u.add_patch(Polygon(poly,closed=True,facecolor=cm.RdYlGn(min(max(a/1.5,0),1)),alpha=0.7))
#         else:
#             ax_u.text(u_min[0]+0.2,u_min[1]+0.2,"Feasible Region = 0",color="red",weight='bold')
#
#         ax_u.scatter(u[0],u[1],c='red',s=60)
#         ax_u.text(u_min[0]+0.2, u_max[1]-0.3, f"Area={a:.3f}", fontsize=11, weight='bold')
#
#     return []

# def update(frame):
#     t = frame * dt
#     times.append(t)
#
#     for i, lam in enumerate(lambda_values):
#         ax_w, ax_u = axs[i]
#
#         # --- QP step ---
#         u, A, B, feasible = qp_control(states[i], t, lam)
#
#         # --- Compute feasible region & its area ---
#         poly = feas_poly(A, B)
#         a = area(poly)
#         areas[i].append(a)
#
#         # === COLLAPSE CONDITION (Feasible set == 0) ===
#         collision = (not feasible) or (a <= 1e-6)
#
#         # --- Integrate dynamics ONLY IF no collision ---
#         x, y, v, th = states[i]
#         if not collision:
#             x += dt * v * np.cos(th)
#             y += dt * v * np.sin(th)
#             v += dt * u[0]
#             th += dt * u[1]
#         # else: freeze state (no change)
#
#         states[i] = np.array([x, y, v, wrap(th)])
#         traj[i].append([x, y])
#
#         # =======================
#         # Workspace Plot
#         # =======================
#         ax_w.clear()
#         ax_w.set_xlim(-5,5); ax_w.set_ylim(-5,5); ax_w.set_aspect('equal')
#         ax_w.set_title(f"Workspace (λ={lam})")
#
#         for (cx,cy,Rb,Re),col in zip(ob_pos(t),obs_colors):
#             ax_w.add_patch(Circle((cx,cy), Re, color=col, alpha=0.75))
#             ax_w.add_patch(Circle((cx,cy), Rb, fill=False,
#                                   linestyle='--', edgecolor=col, linewidth=2))
#
#         if len(traj[i]) > 2:
#             tr = np.array(traj[i])
#             ax_w.plot(tr[:,0], tr[:,1], 'g', linewidth=3)
#
#         ax_w.scatter(goal[0],goal[1],c='black',s=80)
#         ax_w.scatter(x,y,c='red',s=70)
#
#         # ---- COLLISION TEXT ----
#         if collision:
#             ax_w.text(-4.5, 4.3, "⚠ COLLISION / NO SAFE CONTROL",
#                       fontsize=14, weight='bold', color='crimson')
#
#         # =======================
#         # Control-Space Plot
#         # =======================
#         ax_u.clear()
#         ax_u.set_xlim(u_min[0],u_max[0]); ax_u.set_ylim(u_min[1],u_max[1])
#         ax_u.set_aspect('equal')
#         ax_u.set_title(f"Feasible Set (λ={lam})")
#         ax_u.grid(True, linestyle=":", linewidth=1)
#
#         box = np.array([
#             [u_min[0],u_min[1]],[u_min[0],u_max[1]],
#             [u_max[0],u_max[1]],[u_max[0],u_min[1]],[u_min[0],u_min[1]]
#         ])
#         ax_u.plot(box[:,0], box[:,1], 'k--', linewidth=3)
#
#         for (ai,bi), col in zip(zip(A,B), obs_colors):
#             ln = halfspace_line(ai, bi)
#             if ln is not None:
#                 ax_u.plot(ln[:,0], ln[:,1], color=col, linewidth=2)
#
#         if poly.shape[0] >= 3:
#             ax_u.add_patch(Polygon(poly, closed=True,
#                                    facecolor=cm.RdYlGn(min(max(a/1.5,0),1)),
#                                    alpha=0.7, edgecolor='black'))
#         else:
#             ax_u.text(u_min[0]+0.2, u_min[1]+0.3,
#                       "Region = 0", color="red", weight='bold')
#
#         # Keep showing the chosen control value, even if robot stops
#         ax_u.scatter(u[0],u[1],c='red',s=60)
#         ax_u.text(u_min[0]+0.2, u_max[1]-0.3,
#                   f"Area={a:.3f}", fontsize=11, weight='bold')
#
#     return []
lambda_values = [0.8, 2.0]
frozen = [False for _ in lambda_values]
xi = 1
def update(frame):
    t = frame * dt
    times.append(t)

    for i, lam in enumerate(lambda_values):
        ax_w, ax_u = axs[i]

        # If this case has already collided → DO NOT UPDATE STATES
        if frozen[i]:
            # Just re-draw frozen frame
            x,y,v,th = states[i]
            # Workspace
            ax_w.clear()
            ax_w.set_xlim(-5,5); ax_w.set_ylim(-5,5); ax_w.set_aspect('equal')
            #ax_w.set_title(f"Workspace (λ={lam})")
            ax_w.set_title(rf"$\alpha(h)=\eta\,h^{{\xi}},\; \eta={lam**2:.2f},\;\xi={xi}$")
            for (cx,cy,Rb,Re),col in zip(ob_pos(t),obs_colors):
                ax_w.add_patch(Circle((cx,cy),Re,color=col,alpha=0.75))
                ax_w.add_patch(Circle((cx,cy),Rb,fill=False,linestyle='--',edgecolor=col,linewidth=2))
            if len(traj[i])>1:
                tr=np.array(traj[i])
                ax_w.plot(tr[:,0],tr[:,1],'g',linewidth=3)
            ax_w.scatter(goal[0],goal[1],c='black',s=80)
            ax_w.scatter(x,y,c='red',s=90)
            ax_w.text(-4.7,4.3,r"COLLISION — NO SAFE CONTROL", color="crimson",
                      weight="bold", fontsize=14)

            # Control-Space
            ax_u.clear()
            ax_u.set_xlim(u_min[0],u_max[0]); ax_u.set_ylim(u_min[1],u_max[1])
            ax_u.set_aspect('equal')
            #ax_u.set_title(f"Feasible Set (λ={lam})")
            ax_u.set_title(rf"Feasible Set  ($\alpha(h)= {lam**2:.2f} h^{xi}$)")
            ax_u.grid(True,linestyle=":",linewidth=1)
            box=np.array([[u_min[0],u_min[1]],[u_min[0],u_max[1]],
                          [u_max[0],u_max[1]],[u_max[0],u_min[1]],[u_min[0],u_min[1]]])
            ax_u.plot(box[:,0],box[:,1],'k--',linewidth=3)
            ax_u.text(u_min[0]+0.2,u_min[1]+0.3,r"Region = 0",color="red",weight='bold',fontsize=14)
            continue   # <<< MOVE TO NEXT λ CASE

        # Otherwise, simulate normally
        u,A,B,feasible = qp_control(states[i], t, lam)
        poly = feas_poly(A, B)
        a = area(poly)
        areas[i].append(a)

        # ===== COLLISION TRIGGER =====
        collision = (not feasible) or (a <= 1e-6)

        if collision:
            frozen[i] = True  # <<< PERMANENT FREEZE
            continue          # <<< DO NOT UPDATE STATE THIS FRAME

        # Integrate dynamics (only if not frozen or collided)
        x,y,v,th = states[i]
        x+=dt*v*np.cos(th); y+=dt*v*np.sin(th)
        v+=dt*u[0]; th+=dt*u[1]
        states[i]=np.array([x,y,v,wrap(th)])
        traj[i].append([x,y])

        # ===== Draw Workspace =====
        ax_w.clear()
        ax_w.set_xlim(-5,5); ax_w.set_ylim(-5,5); ax_w.set_aspect('equal')
        #ax_w.set_title(f"Workspace (λ={lam})")
        ax_w.set_title(rf"Workspace $\alpha(h)=\eta\,h^{{\xi}},\; \eta={lam ** 2:.2f},\;\xi={xi}$")
        for (cx,cy,Rb,Re),col in zip(ob_pos(t),obs_colors):
            ax_w.add_patch(Circle((cx,cy),Re,color=col,alpha=0.75))
            ax_w.add_patch(Circle((cx,cy),Rb,fill=False,
                                  linestyle='--',edgecolor=col,linewidth=2))
        if len(traj[i])>1:
            tr=np.array(traj[i])
            ax_w.plot(tr[:,0],tr[:,1],'g',linewidth=3)
        ax_w.scatter(goal[0],goal[1],c='black',s=80)
        ax_w.scatter(x,y,c='red',s=70)

        # ===== Draw Control Space =====
        ax_u.clear()
        ax_u.set_xlim(u_min[0],u_max[0]); ax_u.set_ylim(u_min[1],u_max[1])
        ax_u.set_aspect('equal')
        #ax_u.set_title(f"Feasible Set (λ={lam})")
        ax_u.set_title(rf"Feasible Set  ($\alpha(h)= {lam**2:.2f} h^{xi}$)")
        ax_u.grid(True,linestyle=":",linewidth=1)

        box=np.array([[u_min[0],u_min[1]],[u_min[0],u_max[1]],
                      [u_max[0],u_max[1]],[u_max[0],u_min[1]],[u_min[0],u_min[1]]])
        ax_u.plot(box[:,0],box[:,1],'k--',linewidth=3)

        for (ai,bi),col in zip(zip(A,B),obs_colors):
            ln = halfspace_line(ai,bi)
            if ln is not None:
                ax_u.plot(ln[:,0],ln[:,1],color=col,linewidth=2)

        if poly.shape[0]>=3:
            ax_u.add_patch(Polygon(poly,closed=True,
                                   facecolor=cm.RdYlGn(min(max(a/1.5,0),1)),
                                   alpha=0.7,edgecolor='black'))
        ax_u.scatter(u[0],u[1],c='red',s=60)
        ax_u.text(u_min[0]+0.2, u_max[1]-0.3, f"Area={a:.3f}", fontsize=15, weight='bold')

    return []


ani = animation.FuncAnimation(fig, update, frames=T, interval=60)

# ---------- SAVE THE ANIMATION ----------
#print("Saved: unicycle_CBF_classK_comparison.mp4")
ani.save("unicycle_CBF_classK_comparison.gif",
         writer=PillowWriter(fps=20),
         savefig_kwargs={'bbox_inches':'tight', 'pad_inches':0})

plt.show()
# ani.save("unicycle_CBF_classK_comparison.mp4", writer="ffmpeg", fps=20)

