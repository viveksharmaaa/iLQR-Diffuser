# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:09:01 2024

@author: Jean-Baptiste Bouvier

Quadcopter environment modified from John Viljoen's
https://github.com/johnviljoen/231A_project
"""

import torch
import numpy as np

from Quadcopter.plots import plot_traj, traj_comparison

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
    
    
    def __init__(self, reset_noise_scale:float = 1e-2, dt: float = 0.01,
                 cylinder_radii = [0.7, 0.7]):
        
        self.name = "Quadcopter"
        self.state_size = 17
        self.action_size = 4
        self.action_min = np.array([[-1., -1., -1., -1.]])
        self.action_max = np.array([[ 1.,  1.,  1.,  1.]])
        self.position_states = [0,1,2,3,4,5,6]
        self.velocity_states = [7,8,9,10,11,12,13,14,15,16]
        
        ### Obstacles: cylinders along the z-axis
        self.target_position = np.array([7., 0., 0.])
        self.N_cylinders = 2 # 2 cylinders
        self.cylinder_radii = cylinder_radii # radius of the cylinders
        self.cylinder_xc = [2.5, 5.2] # cylinders x center position
        self.cylinder_yc = [0.5, -0.5] # cylinders y center position
        
        
        ### Fundamental quad parameters
        self.g = 9.81 # gravity (m/s^2)
        self.mB = 1.2 # mass (kg)
        self.dxm = 0.16 # arm length (m)
        self.dym =  0.16 # arm length (m)
        self.dzm = 0.01  # arm height (m)
        self.IB = np.array([[0.0123, 0,      0     ],
                            [0,      0.0123, 0     ],
                            [0,      0,      0.0224]])  # Inertial tensor (kg*m^2)
        self.IRzz = 2.7e-5  # rotor moment of inertia (kg*m^2)
        self.Cd = 0.1  # drag coefficient (omnidirectional)
        self.kTh = 1.076e-5  # thrust coeff (N/(rad/s)^2)  (1.18e-7 N/RPM^2)
        self.kTo = 1.632e-7  # torque coeff (Nm/(rad/s)^2)  (1.79e-9 Nm/RPM^2)
        self.minThr = 0.1*4  # Minimum total thrust (N)
        self.maxThr = 9.18*4  # Maximum total thrust (N)
        self.minWmotor = 75  # Minimum motor rotation speed (rad/s)
        self.maxWmotor = 925  # Maximum motor rotation speed (rad/s)
        self.tau = 0.015  # Value for second order system for Motor dynamics
        self.kp = 1.0  # Value for second order system for Motor dynamics
        self.damp = 1.0  # Value for second order system for Motor dynamics
        self.usePrecession = True  # model precession or not
        self.w_hover = 522.9847140714692 # hardcoded hover rotor speed (rad/s)
        
        ### post init useful parameters for quad
        self.B0 = np.array([[self.kTh, self.kTh, self.kTh, self.kTh],
                            [self.dym * self.kTh, - self.dym * self.kTh, -self.dym * self.kTh,  self.dym * self.kTh],
                            [self.dxm * self.kTh,  self.dxm * self.kTh, -self.dxm * self.kTh, -self.dxm * self.kTh],
                            [-self.kTo, self.kTo, - self.kTo, self.kTo]]) # actuation matrix

        self.low_bound = np.array([-100, -100, -100, *[-np.inf]*4, *[-100]*3, *[-100]*3, *[self.minWmotor]*4])
                               # xyz       q0123         xdydzd    pdqdrd    w0123
        
        self.high_bound = np.array([100, 100, 100, *[np.inf]*4, *[100]*3, *[100]*3, *[self.maxWmotor]*4])
                                # xyz      q0123        xdydzd   pdqdrd   w0123
        
        self.dt = dt # time step
        self.reset_noise = reset_noise_scale
        self.hover_state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, *[522.9847140714692]*4])
        
        
    def reset(self):
        self.state = self.hover_state.copy() + self.reset_noise*np.random.randn(self.state_size)
        return self.state.copy()
        
    
    def reset_to(self, state):
        # assert all(state > self.x_lb) and all(state < self.x_ub), "state is out of bounds x_lb, x_ub"
        self.state = state.copy()
        return state


    def is_in_obstacle(self, x, y):
        """Checks whether (x,y) position is inside an obstacle"""
        for i in range(self.N_cylinders):
            inside = self.cylinder_radii[i]**2 >= (x - self.cylinder_xc[i])**2 + (y - self.cylinder_yc[i])**2 
            if inside:
                return True
        return False
    

    def step(self, u):
        
        q0 =    self.state[3]
        q1 =    self.state[4]
        q2 =    self.state[5]
        q3 =    self.state[6]
        xdot =  self.state[7]
        ydot =  self.state[8]
        zdot =  self.state[9]
        p =     self.state[10]
        q =     self.state[11]
        r =     self.state[12]
        wM1 =   self.state[13]
        wM2 =   self.state[14]
        wM3 =   self.state[15]
        wM4 =   self.state[16]
    
        # instantaneous thrusts and torques generated by the current w0...w3
        wMotor = np.stack([wM1, wM2, wM3, wM4])
        wMotor = np.clip(wMotor, self.minWmotor, self.maxWmotor) # this clip shouldn't occur within the dynamics
        th = self.kTh * wMotor ** 2 # thrust
        to = self.kTo * wMotor ** 2 # torque

        def smooth_sign(x, alpha=50.0):
            return -np.tanh(alpha * x)
    
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
                (self.Cd * smooth_sign(-xdot) * xdot**2 #np.sign(-xdot)
                    - 2 * (q0 * q2 + q1 * q3) * (th[0] + th[1] + th[2] + th[3])
                )
                /  self.mB, # xdd
                (
                     self.Cd * smooth_sign(-ydot) * ydot**2 #np.sign(-ydot)
                    + 2 * (q0 * q1 - q2 * q3) * (th[0] + th[1] + th[2] + th[3])
                )
                /  self.mB, # ydd
                (
                    - self.Cd * smooth_sign(zdot) * zdot**2 #np.sign(zdot)
                    - (th[0] + th[1] + th[2] + th[3])
                    * (q0**2 - q1**2 - q2**2 + q3**2)
                    + self.g *  self.mB
                )
                /  self.mB, # zdd (the - in front turns increased height to be positive - SWU)
                (
                    ( self.IB[1,1] -  self.IB[2,2]) * q * r
                    -  self.usePrecession *  self.IRzz * (wM1 - wM2 + wM3 - wM4) * q
                    + (th[0] - th[1] - th[2] + th[3]) *  self.dym
                )
                /  self.IB[0,0], # pd
                (
                    ( self.IB[2,2] -  self.IB[0,0]) * p * r
                    +  self.usePrecession *  self.IRzz * (wM1 - wM2 + wM3 - wM4) * p
                    + (th[0] + th[1] - th[2] - th[3]) *  self.dxm
                )
                /  self.IB[1,1], #qd
                (( self.IB[0,0] -  self.IB[1,1]) * p * q - to[0] + to[1] - to[2] + to[3]) /  self.IB[2,2], # rd
                u[0]/self.IRzz, u[1]/self.IRzz, u[2]/self.IRzz, u[3]/self.IRzz # w0d ... w3d
            ]
        )
    
        self.state += xd * self.dt # one time step forward
        # Clip the rotor speeds within limits
        self.state[13:17] = np.clip(self.state[13:17], self.low_bound[13:17], self.high_bound[13:17])
        
        out_of_bound = any(self.state < self.low_bound) or any(self.state > self.high_bound) # out of bound state
        collided = self.is_in_obstacle(self.state[0], self.state[1])
        distance_to_target = np.linalg.norm(self.state[:3] - self.target_position)
        reward = 1 - out_of_bound - collided - distance_to_target
        terminated = out_of_bound or collided
    
        return self.state.copy(), reward, terminated, False, None


    # Function called by the projectors
    def pos_from_vel(self, S_t, vel_t_dt):
        """
        Calculates the next state's position using explicit Euler integrator
        and quaternion formula, does NOT need to know the dynamics.
        
        Arguments:
            - S_t : current state torch.tensor (17,)
            - vel_t_dt : (unused) next state's velocity torch.tensor (10,)
        Returns:
            - x_t_dt : next state's position torch.tensor (7,)
        """
        x_t_dt = S_t[:7].clone() # copy the current position
        q0 =    S_t[3]
        q1 =    S_t[4]
        q2 =    S_t[5]
        q3 =    S_t[6]
        xdot =  S_t[7]
        ydot =  S_t[8]
        zdot =  S_t[9]
        p =     S_t[10]
        q =     S_t[11]
        r =     S_t[12]
        
        x_t_dt += self.dt*torch.FloatTensor([xdot, ydot, zdot,
                                   -0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r,
                                    0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r,
                                    0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r,
                                   -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r]).to(S_t.device)

        return x_t_dt

    # Plotting functions
    def plot_traj(self, Traj, title:str = ""):
        """Plots the xy trajectory of the Quadcopter."""
        plot_traj(self, Traj, title,'--')

    
    def traj_comparison(self, traj_1, label_1, traj_2, label_2, title:str = "",
                        traj_3=None, label_3=None, traj_4=None, label_4=None,
                        legend_loc='best'):
        """
        Compares up to 4 xy trajectories of the Quadcopter
        Arguments:
            - traj_1 : first trajectory of shape (H, 17)
            - label_1 : corresponding label to display
            - traj_2 : first trajectory of shape (H, 17)
            - label_2 : corresponding label to display
            - title: optional title of the plot
            - traj_3 : optional third trajectory of shape (H, 17)
            - label_3 : optional corresponding label to display
            - traj_4 : optional fourth trajectory of shape (H, 17)
            - label_4 : optional corresponding label to display
            - legend_loc : optional location of the legend
        """
        traj_comparison(self, traj_1, label_1, traj_2, label_2, title,
                        traj_3, label_3, traj_4, label_4, legend_loc)

    import numpy as np

    def quad_analytic_jacobian(self,u):
        """
        Analytic Jacobian A=∂xd/∂x, B=∂xd/∂u for your 'xd' (the derivative vector).
        state ∈ R^17 with indices:
          0:x, 1:y, 2:z, 3:q0, 4:q1, 5:q2, 6:q3,
          7:xdot, 8:ydot, 9:zdot, 10:p, 11:q, 12:r,
          13:w1, 14:w2, 15:w3, 16:w4
        u ∈ R^4 maps to [w1dot, w2dot, w3dot, w4dot] via u/IRzz.
        # p is a dict of constants: kTh, kTo, mB, g, Cd, dym, dxm, IRzz, IB (3x3),
        #                            min/maxWmotor (unused in Jacobian), usePrecession (0/1)
        Returns:
          A (17x17), B (17x4)
        """
        # Unpack state
        x, y, z, q0, q1, q2, q3, xdot, ydot, zdot, p_body, q_body, r_body, w1, w2, w3, w4 = self.state
        # u1, u2, u3, u4 = u

        # Unpack params
        kTh = self.kTh
        kTo = self.kTo
        mB = self.mB
        g = self.g
        Cd = self.Cd
        dym = self.dym
        dxm = self.dxm
        IRzz = self.IRzz
        IB = self.IB
        usePrec = self.usePrecession

        # Helpful quantities
        w = np.array([w1, w2, w3, w4])
        th = kTh * w ** 2  # per-rotor thrust
        to = kTo * w ** 2  # per-rotor torque
        S = th.sum()  # total thrust

        # Signs for drag (a.e. derivative; define at 0 as 0)
        sx = 0.0 if xdot == 0 else np.sign(xdot)
        sy = 0.0 if ydot == 0 else np.sign(ydot)
        sz = 0.0 if zdot == 0 else np.sign(zdot)

        # Precompute inertia scalars
        IBxx, IByy, IBzz = IB[0, 0], IB[1, 1], IB[2, 2]
        a_x = (IByy - IBzz) / IBxx
        a_y = (IBzz - IBxx) / IByy
        a_z = (IBxx - IByy) / IBzz

        # Initialize Jacobians
        A = np.zeros((17, 17))
        B = np.zeros((17, 4))

        # -----------------------------
        # Kinematics: xdot, ydot, zdot
        # -----------------------------
        A[0, 7] = 1.0  # d(xdot)/d(xdot)
        A[1, 8] = 1.0
        A[2, 9] = 1.0

        # ---------------------------------------
        # Quaternion kinematics (omega to qdot)
        # q0dot = -0.5*(p*q1 + q*q2 + r*q3)
        # q1dot =  0.5*(p*q0 - q*q3 + r*q2)
        # q2dot =  0.5*(p*q3 + q*q0 - r*q1)
        # q3dot =  0.5*(-p*q2 + q*q1 + r*q0)
        # ---------------------------------------
        # ∂q0dot/∂q1 = -0.5 p, ∂q0dot/∂q2 = -0.5 q, ∂q0dot/∂q3 = -0.5 r
        A[3, 4] = -0.5 * p_body
        A[3, 5] = -0.5 * q_body
        A[3, 6] = -0.5 * r_body
        # ∂q0dot/∂p = -0.5 q1, ∂/∂q = -0.5 q2, ∂/∂r = -0.5 q3
        A[3, 10] = -0.5 * q1
        A[3, 11] = -0.5 * q2
        A[3, 12] = -0.5 * q3

        # q1dot
        A[4, 3] = 0.5 * p_body
        A[4, 6] = 0.5 * r_body
        A[4, 5] = -0.5 * q_body
        A[4, 10] = 0.5 * q0
        A[4, 11] = -0.5 * q3
        A[4, 12] = 0.5 * q2

        # q2dot
        A[5, 6] = 0.5 * p_body
        A[5, 3] = 0.5 * q_body
        A[5, 4] = -0.5 * r_body
        A[5, 10] = 0.5 * q3
        A[5, 11] = 0.5 * q0
        A[5, 12] = -0.5 * q1

        # q3dot
        A[6, 5] = -0.5 * p_body
        A[6, 4] = 0.5 * q_body
        A[6, 3] = 0.5 * r_body
        A[6, 10] = -0.5 * q2
        A[6, 11] = 0.5 * q1
        A[6, 12] = 0.5 * q0

        # -----------------------------------
        # Translational accelerations (xdd)
        # xdd = ( Cd*sign(-xdot)*xdot^2 - 2(q0 q2 + q1 q3) * S ) / mB
        # ydd = ( Cd*sign(-ydot)*ydot^2 + 2(q0 q1 - q2 q3) * S ) / mB
        # zdd = ( -Cd*sign(zdot)*zdot^2
        #         - S*(q0^2 - q1^2 - q2^2 + q3^2) + g*mB ) / mB
        # -----------------------------------

        # Drag derivatives (a.e.): d/dv [Cd*sign(-v)*v^2] = -2*Cd*|v|
        A[7 + 1, 7] = (-2.0 * Cd * abs(xdot)) / mB  # row 8 wrt xdot (xdd w.r.t xdot)
        A[8 + 1, 8] = (-2.0 * Cd * abs(ydot)) / mB  # row 9 wrt ydot
        A[9 + 1, 9] = (-2.0 * Cd * abs(zdot)) / mB  # row 10 wrt zdot, with the leading minus already in zdd

        # Shorthands for quaternion combos
        c1 = -2.0 * (q0 * q2 + q1 * q3)  # coefficient in xdd multiplying S
        c2 = 2.0 * (q0 * q1 - q2 * q3)  # coefficient in ydd multiplying S
        c3 = -(q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)  # coefficient in zdd multiplying S

        # ∂xdd/∂q*
        A[8, 3] = (-2.0 * q2) * S / mB
        A[8, 5] = (-2.0 * q0) * S / mB
        A[8, 4] = (-2.0 * q3) * S / mB
        A[8, 6] = (-2.0 * q1) * S / mB

        # ∂ydd/∂q*
        A[9, 3] = (2.0 * q1) * S / mB
        A[9, 4] = (2.0 * q0) * S / mB
        A[9, 5] = (-2.0 * q3) * S / mB
        A[9, 6] = (-2.0 * q2) * S / mB

        # ∂zdd/∂q*
        A[10, 3] = (-2.0 * q0) * S / mB
        A[10, 4] = (2.0 * q1) * S / mB
        A[10, 5] = (2.0 * q2) * S / mB
        A[10, 6] = (-2.0 * q3) * S / mB

        # ∂xdd/∂w_i via S (S = kTh * sum(2*w_i*w_i?) → dS/dw_i = 2*kTh*w_i)
        dSdw = 2.0 * kTh * w
        A[8, 13:17] = (c1 / mB) * dSdw
        A[9, 13:17] = (c2 / mB) * dSdw
        A[10, 13:17] = (c3 / mB) * dSdw

        # -----------------------------------
        # Angular accelerations (p,q,r) dot
        # pd = [ a_x*q*r - usePrec*IRzz*(w1-w2+w3-w4)*q + (th1 - th2 - th3 + th4)*dym ] / IBxx
        # qd = [ a_y*p*r + usePrec*IRzz*(w1-w2+w3-w4)*p + (th1 + th2 - th3 - th4)*dxm ] / IByy
        # rd = [ a_z*p*q - to1 + to2 - to3 + to4 ] / IBzz
        # -----------------------------------

        # ∂pd/∂q, ∂pd/∂r
        A[11, 11] = (a_x * r_body) / IBxx
        A[11, 12] = (a_x * q_body) / IBxx

        # ∂pd/∂w_i from gyro (linear in w) and thrust-asymmetry (through th = kTh w^2)
        s_gyro = np.array([1, -1, 1, -1], dtype=float)  # (w1 - w2 + w3 - w4)
        d_pd_dwi_gyro = (-usePrec * IRzz * q_body / IBxx) * s_gyro
        d_pd_dwi_th = (dym / IBxx) * (2.0 * kTh * w) * np.array([1, -1, -1, 1], float)
        A[11, 13:17] = d_pd_dwi_gyro + d_pd_dwi_th

        # qd partials
        A[12, 10] = (a_y * r_body) / IByy
        A[12, 12] = (a_y * p_body) / IByy
        d_qd_dwi_gyro = (usePrec * IRzz * p_body / IByy) * s_gyro
        d_qd_dwi_th = (dxm / IByy) * (2.0 * kTh * w) * np.array([1, 1, -1, -1], float)
        A[12, 13:17] = d_qd_dwi_gyro + d_qd_dwi_th

        # rd partials
        A[13, 10] = (a_z * q_body) / IBzz
        A[13, 11] = (a_z * p_body) / IBzz
        # torque asymmetry: -to1 + to2 - to3 + to4, to_i = kTo w_i^2 ⇒ d/dw_i = ± 2*kTo*w_i
        A[13, 13:17] = (1.0 / IBzz) * (2.0 * kTo * w) * np.array([-1, +1, -1, +1], float)

        # -----------------------
        # Motor first-order rates
        # wdot_i = u[i]/IRzz  (no state dependence)
        # -----------------------
        B[14, 0] = 1.0 / IRzz
        B[15, 1] = 1.0 / IRzz
        B[16, 2] = 1.0 / IRzz
        B[17 - 1, 3] = 1.0 / IRzz  # index 16 (w4dot)

        # All other B entries = 0
        # A entries for positions already set; remaining rows (x,y,z) have no direct state deps.

        return A, B

         











