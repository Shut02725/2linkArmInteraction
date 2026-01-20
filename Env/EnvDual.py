import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from collections import deque
from target_integration.partner_state_estimation import partner_estimation

class DualTrackingEnv():
    def __init__(self, dt=0.01, noise_config=None):
        self.dt = dt                # time step (s)
        self.time = 0               # simulation time (s)
        self.step_count = 0         # step count
        self.delay_steps = 12       # control delay steps (steps)
        self.m = 1.0                # mass of the end-effector for partner estimation
        
        # parameter of 2link arm (Agent 1)
        self.agent1_L_s = torch.tensor(0.3, dtype=torch.float32, device=device)  # link length (m)
        self.agent1_L_e = torch.tensor(0.35, dtype=torch.float32, device=device) # link length (m)
        self.agent1_m_s = torch.tensor(1.9, dtype=torch.float32, device=device)  # mass (kg)
        self.agent1_m_e = torch.tensor(1.7, dtype=torch.float32, device=device)  # mass (kg)
        self.agent1_I_s = torch.tensor(0.05, dtype=torch.float32, device=device) # Inertia (kg*m^2)
        self.agent1_I_e = torch.tensor(0.06, dtype=torch.float32, device=device) # Inertia (kg*m^2)
        
        # parameter of 2link arm (Agent 2)
        self.agent2_L_s = torch.tensor(0.3, dtype=torch.float32, device=device)  # link length (m)
        self.agent2_L_e = torch.tensor(0.35, dtype=torch.float32, device=device) # link length (m)
        self.agent2_m_s = torch.tensor(1.9, dtype=torch.float32, device=device)  # mass (kg)
        self.agent2_m_e = torch.tensor(1.7, dtype=torch.float32, device=device)  # mass (kg)
        self.agent2_I_s = torch.tensor(0.05, dtype=torch.float32, device=device) # Inertia (kg*m^2)
        self.agent2_I_e = torch.tensor(0.06, dtype=torch.float32, device=device) # Inertia (kg*m^2)
        
        # initial joint angle setting
        q_s_init_rad = torch.tensor(torch.pi / 4, dtype=torch.float32, device=device)  # 45deg
        q_e_init_rad = torch.tensor(torch.pi / 2, dtype=torch.float32, device=device)  # 90deg
        initial_q = torch.tensor([q_s_init_rad, q_e_init_rad], dtype=torch.float32, device=device)
        
        # initial position setting (Agent1)
        agent1_x_init_shoulder = - (self.agent1_L_s * torch.cos(q_s_init_rad) + self.agent1_L_e * torch.cos(q_s_init_rad + q_e_init_rad))
        agent1_y_init_shoulder = - (self.agent1_L_s * torch.sin(q_s_init_rad) + self.agent1_L_e * torch.sin(q_s_init_rad + q_e_init_rad))
        agent1_x_init_elbow = - (self.agent1_L_e * torch.cos(q_s_init_rad + q_e_init_rad))
        agent1_y_init_elbow = - (self.agent1_L_e * torch.sin(q_s_init_rad + q_e_init_rad))
        
        # initial position setting (Agent2)
        agent2_x_init_shoulder = - (self.agent2_L_s * torch.cos(q_s_init_rad) + self.agent2_L_e * torch.cos(q_s_init_rad + q_e_init_rad))
        agent2_y_init_shoulder = - (self.agent2_L_s * torch.sin(q_s_init_rad) + self.agent2_L_e * torch.sin(q_s_init_rad + q_e_init_rad))
        agent2_x_init_elbow = - (self.agent2_L_e * torch.cos(q_s_init_rad + q_e_init_rad))
        agent2_y_init_elbow = - (self.agent2_L_e * torch.sin(q_s_init_rad + q_e_init_rad))
        
        # Noise configuration
        if noise_config is None:
            noise_config = {
                'visual_noise_pos_std': 0.0,           # 視覚ノイズ（位置）
                'visual_noise_vel_std': 0.0,           # 視覚ノイズ (速度 %)
                'visual_noise_acc_std': 0.0,           # 視覚ノイズ (加速度 %)
                'haptic_noise_std': 0.0,           # 触覚（力）- 信号依存
                'motor_noise_std': 0.0,             # 運動指令ノイズ - 信号依存
                'enable_noise': True
            }
        self.noise_config = noise_config

        # Target initialization
        self.target_pos = torch.zeros(2, device=device)
        self.target_vel = torch.zeros(2, device=device)
        self.target_acc = torch.zeros(2, device=device)
        self.target_pos_noisy = torch.zeros(2, device=device)
        self.target_vel_noisy = torch.zeros(2, device=device)
        self.target_acc_noisy = torch.zeros(2, device=device)

        # Agent 1 initialization
        self.agent1_pos_end = torch.zeros(2, device=device)
        self.agent1_pos_base = torch.tensor([agent1_x_init_shoulder, agent1_y_init_shoulder], device=device)
        self.agent1_vel = torch.zeros(2, device=device)
        self.agent1_acc = torch.zeros(2, device=device)
        self.agent1_q = torch.tensor([q_s_init_rad, q_e_init_rad], device=device)
        self.agent1_q_dot = torch.zeros(2, device=device)
        self.agent1_q_ddot = torch.zeros(2, device=device)
        self.agent1_control = torch.zeros(2, device=device)
        self.agent1_control_buffer = deque(maxlen=self.delay_steps)
        self.agent1_force = torch.zeros(2, device=device)
        self.agent1_tau = torch.zeros(2, device=device)
        self.agent1_pos_noisy = torch.zeros(2, device=device)
        self.agent1_vel_noisy = torch.zeros(2, device=device)
        self.agent1_acc_noisy = torch.zeros(2, device=device)
        self.agent1_control_noisy = torch.zeros(2, device=device)

        # Agent 2 initialization
        self.agent2_pos_end = torch.zeros(2, device=device)
        self.agent2_pos_base = torch.tensor([agent2_x_init_shoulder, agent2_y_init_shoulder], device=device)
        self.agent2_vel = torch.zeros(2, device=device)
        self.agent2_acc = torch.zeros(2, device=device)
        self.agent2_q = torch.tensor([q_s_init_rad, q_e_init_rad], device=device)
        self.agent2_q_dot = torch.zeros(2, device=device)
        self.agent2_q_ddot = torch.zeros(2, device=device)
        self.agent2_control = torch.zeros(2, device=device)
        self.agent2_control_buffer = deque(maxlen=self.delay_steps)
        self.agent2_force = torch.zeros(2, device=device)
        self.agent2_tau = torch.zeros(2, device=device)
        self.agent2_pos_noisy = torch.zeros(2, device=device)
        self.agent2_vel_noisy = torch.zeros(2, device=device)
        self.agent2_acc_noisy = torch.zeros(2, device=device)
        self.agent2_control_noisy = torch.zeros(2, device=device)

        # Interaction force
        self.F_interaction = torch.zeros(2, device=device) 
        self.F_interaction_noisy = torch.zeros(2, device=device)
        self.k_interaction = 120
        self.c_interaction = 7
        
        # partner estimation
        n_obs_partner = 14

        Q = 50 * torch.tensor([[self.dt**6 / 36, self.dt**5 / 12, self.dt**4 / 6],
                                [self.dt**5 / 12, self.dt**4 / 4, self.dt**3 / 2],
                                [self.dt**4 / 6, self.dt**3 / 2, self.dt**2]], dtype=torch.float32, device=device)

        Q_p = torch.zeros((26, 26), dtype=torch.float32, device=device)
        Q_p[0:3, 0:3] = Q
        Q_p[3:6, 3:6] = Q
        Q_p[6:9, 6:9] = Q
        Q_p[9:12, 9:12] = Q
        Q_p[12:15, 12:15] = Q
        Q_p[15:18, 15:18] = Q
        Q_p[18:20, 18:20] = torch.eye(2, dtype=torch.float32, device=device)
        Q_p[20:26, 20:26] = 1e-3 * torch.eye(6, dtype=torch.float32, device=device)
        R_p = torch.eye(14, dtype=torch.float32, device=device) * 0.0001

        B_p = torch.tensor([0, 0, self.dt / self.m, 0, 0, self.dt / self.m], dtype=torch.float32, device=device).view(-1, 1)
        H_p = torch.zeros((14, 26), dtype=torch.float32, device=device)
        H_p[0:6, 0:6] = torch.eye(6, dtype=torch.float32, device=device)
        H_p[6:14, 12:20] = torch.eye(8, dtype=torch.float32, device=device)

        self.agent1_partner_estimation = partner_estimation(self.dt, B_p, self.k_interaction, self.c_interaction, Q_p, R_p, H_p)
        self.agent2_partner_estimation = partner_estimation(self.dt, B_p, self.k_interaction, self.c_interaction, Q_p, R_p, H_p)

        self.agent1_partner_obs = torch.zeros(n_obs_partner, device=device)
        self.agent2_partner_obs = torch.zeros(n_obs_partner, device=device)

        # For recording trajectory
        self.trajectory_history = {
            'target': [],
            'agent1': [],
            'agent2': [],
            'agent1_angles': [],
            'agent2_angles': [],
            'time': []
        }

        n_obs_self = 4

        self.agent1_self_obs = torch.zeros(n_obs_self, device=device)
        self.agent2_self_obs = torch.zeros(n_obs_self, device=device)

        self.agent1_pos_error = torch.zeros(2, device=device)
        self.agent1_vel_error = torch.zeros(2, device=device)
        self.agent1_acc_error = torch.zeros(2, device=device)

        self.agent2_pos_error = torch.zeros(2, device=device)
        self.agent2_vel_error = torch.zeros(2, device=device)
        self.agent2_acc_error = torch.zeros(2, device=device)

    def add_noise(self, tensor, noise_std):
        if not self.noise_config['enable_noise'] or noise_std == 0 or torch.isnan(noise_std) or torch.isinf(noise_std):
            return tensor
        noise = torch.normal(0, noise_std, size=tensor.shape, device=device)
        return tensor + noise
    
    def target_traj(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(float(t), device=device)
        # target position
        x = (3*torch.sin(1.8*t) + 3.4*torch.sin(1.9*t) + 2.5*torch.sin(1.82*t) + 4.3*torch.sin(2.34*t)) / 100
        y = (3*torch.sin(1.1*t) + 3.2*torch.sin(3.6*t) + 3.8*torch.sin(2.5*t) + 4.8*torch.sin(1.48*t)) / 100
        # target velocity
        vx = (3*1.8*torch.cos(1.8*t) + 3.4*1.9*torch.cos(1.9*t) + 2.5*1.82*torch.cos(1.82*t) + 4.3*2.34*torch.cos(2.34*t)) / 100
        vy = (3*1.1*torch.cos(1.1*t) + 3.2*3.6*torch.cos(3.6*t) + 3.8*2.5*torch.cos(2.5*t) + 4.8*1.48*torch.cos(1.48*t)) / 100
        # target acceleration
        ax = (-3*1.8*1.8*torch.sin(1.8*t) - 3.4*1.9*1.9*torch.sin(1.9*t) - 2.5*1.82*1.82*torch.sin(1.82*t) - 4.3*2.34*2.34*torch.sin(2.34*t)) / 100
        ay = (-3*1.1*1.1*torch.sin(1.1*t) - 3.2*3.6*3.6*torch.sin(3.6*t) - 3.8*2.5*2.5*torch.sin(2.5*t) - 4.8*1.48*1.48*torch.sin(1.48*t)) / 100
        # return target state
        pos = torch.stack([x, y])
        vel = torch.stack([vx, vy])
        acc = torch.stack([ax, ay])
        return pos, vel, acc

    def forward_kinematics(self, L_s, L_e, q):
        q_s, q_e = q[0], q[1]
        x = L_s * torch.cos(q_s) + L_e * torch.cos(q_s + q_e)
        y = L_s * torch.sin(q_s) + L_e * torch.sin(q_s + q_e)
        return torch.stack([x, y])

    def calc_jacobian_and_dot(self, L_s, L_e, q, dq):
        q_s, q_e = q[0], q[1]
        dq_s, dq_e = dq[0], dq[1]
        c_s, s_s = torch.cos(q_s), torch.sin(q_s)
        c_se, s_se = torch.cos(q_s + q_e), torch.sin(q_s + q_e)
        dq_sum = dq_s + dq_e # q_s + q_e の時間微分
        
        J11 = -L_s * s_s - L_e * s_se
        J12 = -L_e * s_se
        J21 = L_s * c_s + L_e * c_se
        J22 = L_e * c_se
        # Jacobian matrix
        J = torch.stack([torch.stack([J11, J12]), torch.stack([J21, J22])])

        J_dot_11 = -L_s * c_s * dq_s - L_e * c_se * dq_sum
        J_dot_12 = -L_e * c_se * dq_sum
        J_dot_21 = -L_s * s_s * dq_s - L_e * s_se * dq_sum
        J_dot_22 = -L_e * s_se * dq_sum
        # Jacobian Time Derivative matrix
        J_dot = torch.stack([torch.stack([J_dot_11, J_dot_12]), torch.stack([J_dot_21, J_dot_22])])
        return J, J_dot

    def calc_mass_matrix(self, q, I_s, I_e, m_e, L_s, L_e):
        q_e = q[1]
        # Compute Inertia
        M11 = I_s + I_e + m_e*L_s**2 + m_e*L_s*L_e*torch.cos(q_e)
        M12 = 0.5*m_e*L_s*L_e*torch.cos(q_e)
        M21 = M12
        M22 = I_e
        # Inertia Matrix
        M = torch.stack([torch.stack([M11, M12]), torch.stack([M21, M22])])
        return M

    def calc_coriolis(self, q, q_dot, m_e, L_s, L_e):
        q_s, q_e = q[0], q[1]
        q_s_dot, q_e_dot = q_dot[0], q_dot[1]
        # Compute Coriolis Force
        C11 = 0.5*m_e*L_s*L_e*torch.sin(q_e) * (-2*q_e_dot)
        C12 = 0.5*m_e*L_s*L_e*torch.sin(q_e) * (q_e_dot)
        C21 = 0.5*m_e*L_s*L_e*torch.sin(q_e) * (q_s_dot)
        C22 = torch.tensor(0.0, dtype=q.dtype, device=q.device)
        # Coriolis Matrix
        C = torch.stack([torch.stack([C11, C12]), torch.stack([C21, C22])])
        return C

    def forward_dynamics(self, M, C, q_dot, tau):
        M_inv = torch.inverse(M)
        q_ddot = torch.matmul(M_inv, tau - torch.matmul(C, q_dot))
        return q_ddot
    
    def dynamics(self, q, q_dot, tau, L_s, L_e, I_s, I_e, m_e):
        M = self.calc_mass_matrix(q, I_s, I_e, m_e, L_s, L_e)
        C = self.calc_coriolis(q, q_dot, m_e, L_s, L_e)
        q_ddot = self.forward_dynamics(M, C, q_dot, tau)
        return q_dot, q_ddot
    
    def rk4_method(self, q, q_dot, tau, L_s, L_e, I_s, I_e, m_e):
        dt = self.dt
        # k1
        dq_k1, dq_dot_k1 = self.dynamics(q, q_dot, tau, L_s, L_e, I_s, I_e, m_e)
        # k2
        q_k2 = q + dq_k1 * (dt / 2)
        q_dot_k2 = q_dot + dq_dot_k1 * (dt / 2)
        dq_k2, dq_dot_k2 = self.dynamics(q_k2, q_dot_k2, tau, L_s, L_e, I_s, I_e, m_e)
        # k3
        q_k3 = q + dq_k2 * (dt / 2)
        q_dot_k3 = q_dot + dq_dot_k2 * (dt / 2)
        dq_k3, dq_dot_k3 = self.dynamics(q_k3, q_dot_k3, tau, L_s, L_e, I_s, I_e, m_e)
        # k4
        q_k4 = q + dq_k3 * dt
        q_dot_k4 = q_dot + dq_dot_k3 * dt
        dq_k4, dq_dot_k4 = self.dynamics(q_k4, q_dot_k4, tau, L_s, L_e, I_s, I_e, m_e)
        q_next = q + (dq_k1 + 2*dq_k2 + 2*dq_k3 + dq_k4) * (dt / 6)
        q_dot_next = q_dot + (dq_dot_k1 + 2*dq_dot_k2 + 2*dq_dot_k3 + dq_dot_k4) * (dt / 6)
        return q_next, q_dot_next, dq_dot_k4
    
    def Agent1_FBController(self, pos_error, vel_error, acc_error):
        kp = 8
        kd = 4
        ka = 0
        FB_control = - kp*pos_error - kd*vel_error - ka*acc_error
        # FB_control_noisy = self.add_noise(FB_control, self.noise_config['control_noise_std'])

        self.agent1_control_buffer.append(FB_control)

        if len(self.agent1_control_buffer) == self.delay_steps:
            self.agent1_force = self.agent1_control_buffer[0].clone()
        else:
            self.agent1_force = torch.zeros(2, device=device)

    def Agent2_FBController(self, pos_error, vel_error, acc_error):
        kp = 6
        kd = 3
        ka = 0
        FB_control = - kp*pos_error - kd*vel_error - ka*acc_error
        # FB_control_noisy = self.add_noise(FB_control, self.noise_config['control_noise_std'])

        self.agent2_control_buffer.append(FB_control)

        if len(self.agent2_control_buffer) == self.delay_steps:
            self.agent2_force = self.agent2_control_buffer[0].clone()
        else:
            self.agent2_force = torch.zeros(2, device=device)

    def step(self):
        self.target_pos, self.target_vel, self.target_acc = self.target_traj(self.time)
        
        target_vel_noise_std = self.noise_config['visual_noise_vel_std'] * torch.norm(self.target_vel)
        target_acc_noise_std = self.noise_config['visual_noise_acc_std'] * torch.norm(self.target_acc)

        self.target_pos_noisy = self.add_noise(self.target_pos, self.noise_config['visual_noise_pos_std'])
        self.target_vel_noisy = self.add_noise(self.target_vel, target_vel_noise_std)
        self.target_acc_noisy = self.add_noise(self.target_acc, target_acc_noise_std)
        
        agent1_vel_noise_std = self.noise_config['visual_noise_vel_std'] * torch.norm(self.agent1_vel)
        agent1_acc_noise_std = self.noise_config['visual_noise_acc_std'] * torch.norm(self.agent1_acc)
        
        self.agent1_pos_noisy = self.add_noise(self.agent1_pos_end, self.noise_config['visual_noise_pos_std'])
        self.agent1_vel_noisy = self.add_noise(self.agent1_vel, agent1_vel_noise_std)
        self.agent1_acc_noisy = self.add_noise(self.agent1_acc, agent1_acc_noise_std)
        
        agent2_vel_noise_std = self.noise_config['visual_noise_vel_std'] * torch.norm(self.agent2_vel)
        agent2_acc_noise_std = self.noise_config['visual_noise_acc_std'] * torch.norm(self.agent2_acc)
        
        self.agent2_pos_noisy = self.add_noise(self.agent2_pos_end, self.noise_config['visual_noise_pos_std'])
        self.agent2_vel_noisy = self.add_noise(self.agent2_vel, agent2_vel_noise_std)
        self.agent2_acc_noisy = self.add_noise(self.agent2_acc, agent2_acc_noise_std)
        
        # 3. 触覚ノイズ（力情報）- 信号依存
        haptic_noise_std = self.noise_config['haptic_noise_std'] * torch.norm(self.F_interaction)
        self.F_interaction_noisy = self.add_noise(self.F_interaction, haptic_noise_std)
        # partner observation (Agent1)
        self.agent1_partner_obs = torch.stack([
            self.agent1_pos_noisy[0], self.agent1_vel_noisy[0], self.agent1_acc_noisy[0],
            self.agent1_pos_noisy[1], self.agent1_vel_noisy[1], self.agent1_acc_noisy[1],
            self.target_pos_noisy[0], self.target_vel_noisy[0], self.target_acc_noisy[0],
            self.target_pos_noisy[1], self.target_vel_noisy[1], self.target_acc_noisy[1],
            self.F_interaction_noisy[0], self.F_interaction_noisy[1]
        ])
        # partner observation (Agent2)
        self.agent2_partner_obs = torch.stack([
            self.agent2_pos_noisy[0], self.agent2_vel_noisy[0], self.agent2_acc_noisy[0],
            self.agent2_pos_noisy[1], self.agent2_vel_noisy[1], self.agent2_acc_noisy[1],
            self.target_pos_noisy[0], self.target_vel_noisy[0], self.target_acc_noisy[0],
            self.target_pos_noisy[1], self.target_vel_noisy[1], self.target_acc_noisy[1],
            -self.F_interaction_noisy[0], -self.F_interaction_noisy[1]
        ])

        # Partner estimation
        agent1_partner_target_pos = self.agent1_partner_estimation.step(self.agent1_partner_obs)
        agent2_partner_target_pos = self.agent2_partner_estimation.step(self.agent2_partner_obs)

        # Self observation（ノイズのある観測を使用）
        self.agent1_self_obs[0] = self.agent1_pos_noisy[0] - self.target_pos_noisy[0]
        self.agent1_self_obs[1] = self.agent1_pos_noisy[0] - (agent1_partner_target_pos[0] if not torch.isnan(agent1_partner_target_pos[0]) else self.target_pos_noisy[0])
        self.agent1_self_obs[2] = self.agent1_pos_noisy[1] - self.target_pos_noisy[1]
        self.agent1_self_obs[3] = self.agent1_pos_noisy[1] - (agent1_partner_target_pos[1] if not torch.isnan(agent1_partner_target_pos[1]) else self.target_pos_noisy[1])
        
        self.agent2_self_obs[0] = self.agent2_pos_noisy[0] - self.target_pos_noisy[0]
        self.agent2_self_obs[1] = self.agent2_pos_noisy[0] - (agent2_partner_target_pos[0] if not torch.isnan(agent2_partner_target_pos[0]) else self.target_pos_noisy[0])
        self.agent2_self_obs[2] = self.agent2_pos_noisy[1] - self.target_pos_noisy[1]
        self.agent2_self_obs[3] = self.agent2_pos_noisy[1] - (agent2_partner_target_pos[1] if not torch.isnan(agent2_partner_target_pos[1]) else self.target_pos_noisy[1])

        # エラー計算（真の値を使用）
        self.agent1_pos_error = self.agent1_pos_end - self.target_pos
        self.agent1_vel_error = self.agent1_vel - self.target_vel
        self.agent1_acc_error = self.agent1_acc - self.target_acc

        self.agent2_pos_error = self.agent2_pos_end - self.target_pos
        self.agent2_vel_error = self.agent2_vel - self.target_vel
        self.agent2_acc_error = self.agent2_acc - self.target_acc

        # print("agent1_target:", agent1_target_pos_noisy, "agent1_partner_target:", agent1_partner_target_pos)

        self.Agent1_FBController(self.agent1_pos_error, self.agent1_vel_error, self.agent1_acc_error)
        self.Agent2_FBController(self.agent2_pos_error, self.agent2_vel_error, self.agent2_acc_error)

        agent1_control_magnitude = torch.norm(self.agent1_control)
        agent1_control_noise_std = agent1_control_magnitude * self.noise_config['motor_noise_std']
        self.agent1_control_noisy = self.add_noise(self.agent1_control, agent1_control_noise_std)

        agent2_control_magnitude = torch.norm(self.agent2_control)
        agent2_control_noise_std = agent2_control_magnitude * self.noise_config['motor_noise_std']
        self.agent2_control_noisy = self.add_noise(self.agent2_control, agent2_control_noise_std)

        # Physics update (Agent1)
        agent1_total_force = self.agent1_force + self.F_interaction
        J1, J1_dot = self.calc_jacobian_and_dot(self.agent1_L_s, self.agent1_L_e, self.agent1_q, self.agent1_q_dot)
        self.agent1_tau = torch.matmul(J1.T, agent1_total_force)
        self.agent1_q, self.agent1_q_dot, self.agent1_q_ddot = self.rk4_method(self.agent1_q, self.agent1_q_dot, self.agent1_tau, self.agent1_L_s, self.agent1_L_e, self.agent1_I_s, self.agent1_I_e, self.agent1_m_e)

        J1, J1_dot = self.calc_jacobian_and_dot(self.agent1_L_s, self.agent1_L_e, self.agent1_q, self.agent1_q_dot)
        self.agent1_pos_end = self.agent1_pos_base + self.forward_kinematics(self.agent1_L_s, self.agent1_L_e, self.agent1_q)
        self.agent1_vel = torch.matmul(J1, self.agent1_q_dot)
        self.agent1_acc = torch.matmul(J1, self.agent1_q_ddot) + torch.matmul(J1_dot, self.agent1_q_dot)
        
        # Physics update (Agent2)
        agent2_total_force = self.agent2_force - self.F_interaction
        J2, J2_dot = self.calc_jacobian_and_dot(self.agent2_L_s, self.agent2_L_e, self.agent2_q, self.agent2_q_dot)
        self.agent2_tau = torch.matmul(J2.T, agent2_total_force)
        self.agent2_q, self.agent2_q_dot, self.agent2_q_ddot = self.rk4_method(self.agent2_q, self.agent2_q_dot, self.agent2_tau, self.agent2_L_s, self.agent2_L_e, self.agent2_I_s, self.agent2_I_e, self.agent2_m_e)

        J2, J2_dot = self.calc_jacobian_and_dot(self.agent2_L_s, self.agent2_L_e, self.agent2_q, self.agent2_q_dot)
        self.agent2_pos_end = self.agent2_pos_base + self.forward_kinematics(self.agent2_L_s, self.agent2_L_e, self.agent2_q)
        self.agent2_vel = torch.matmul(J2, self.agent2_q_dot)
        self.agent2_acc = torch.matmul(J2, self.agent2_q_ddot) + torch.matmul(J2_dot, self.agent2_q_dot)

        print('agent1_pos:', self.agent1_pos_end.cpu().numpy(), 'target_pos:', self.target_pos.cpu().numpy(), 'pos_error:', self.agent1_pos_error.cpu().numpy())

        self.time += self.dt
        self.step_count += 1
        # Trajectory recording (省略)
        self.trajectory_history['agent1'].append(self.agent1_pos_end.cpu().numpy().copy())
        self.trajectory_history['agent2'].append(self.agent2_pos_end.cpu().numpy().copy())
        self.trajectory_history['target'].append(self.target_pos.cpu().numpy().copy())
        self.trajectory_history['agent1_angles'].append(self.agent1_q.cpu().numpy().copy())
        self.trajectory_history['agent2_angles'].append(self.agent2_q.cpu().numpy().copy())
        self.trajectory_history['time'].append(self.time)