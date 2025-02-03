import logging
import time
from copy import deepcopy
import os

import numpy as np
import pandas as pd
import casadi as cs

from matplotlib import pyplot as plt
import pickle

from utils import mpc_utils

class CrowdAwareMPC:
    def __init__(self, config):
        if config is not None:
            self.configure(config)
        else:
            raise ValueError('Please provide a configuration file')
        
        self.kinematics = 'kinematic'
        
        # mpc solver attributes
        self.nx = 4  # State: [x, y, orientation, speed]
        self.nu = 2  # Control: [a, omega]
        
        # Reference trajectory placeholders
        self.prev_state = None
        self.prev_control = None
        
        # MPC Solver
        self.opti = None
        self.opti_dict = {}
        self.init_solver()

    def configure(self, config):
        self.pref_speed =  config.getfloat('mpc_env', 'pref_speed')
        self.max_speed = config.getfloat('mpc_env', 'max_speed')
        self.max_rev_speed = config.getfloat('mpc_env', 'max_speed')
        self.max_rot = config.getfloat('mpc_env', 'max_rot_degrees') * np.pi / 180.0
        self.max_l_acc = config.getfloat('mpc_env', 'max_l_acc')
        self.max_l_dcc = config.getfloat('mpc_env', 'max_l_dcc')
        
        self.max_human_groups = config.getint('mpc_env', 'max_human_groups')
        self.mpc_horizon = config.getint('mpc_env', 'mpc_horizon')
        # logging.info('[MPCEnv] Config {:} = {:}'.format('pref_speed', self.pref_speed))
        # logging.info('[MPCEnv] Config {:} = {:}'.format('max_speed', self.max_speed))
        # logging.info('[MPCEnv] Config {:} = {:}'.format('max_rev_speed', self.max_rev_speed))
        # logging.info('[MPCEnv] Config {:} = {:}'.format('max_rot', self.max_rot))
        # logging.info('[MPCEnv] Config {:} = {:}'.format('max_l_acc', self.max_l_acc))
        # logging.info('[MPCEnv] Config {:} = {:}'.format('max_l_dcc', self.max_l_dcc))
        
        self.w_goal = config.getfloat('mpc_env', 'w_goal')
        self.w_safe = config.getfloat('mpc_env', 'w_safe')
        self.w_follow = config.getfloat('mpc_env', 'w_follow')
        self.w_smooth = config.getfloat('mpc_env', 'w_smooth')
        
        self.d_safe = config.getfloat('mpc_env', 'd_safe')
        
        self.dt = config.getfloat('mpc_env', 'dt')
        
    def init_solver(self, ipopt_print_level=0):
        """Sets up nonlinear optimization problem.
        Adapted from 
        # (safe-control-gym/safe_control_gym/controllers/mpc/mpc.py)
        """

        opti = cs.Opti()
        x_var = opti.variable(self.nx, self.mpc_horizon + 1)
        u_var = opti.variable(self.nu, self.mpc_horizon)
        
        # Initial state and target
        x_init = opti.parameter(self.nx, 1)
        target = opti.parameter(2, 1)
        
        # crowd-related parameters
        crowd_pos = opti.parameter(6, self.max_human_groups, self.mpc_horizon)
        
        # Cost function
        cost = 0
        for t in range(self.mpc_horizon):
            # Goal tracking
            cost_goal = cs.sumsqr(x_var[:2, t] - target)
            
            # Crowd following
            cost_follow = cs.sumsqr(x_var[:2, t] - crowd_pos)  # Match position
            cost_follow += cs.sumsqr(x_var[3, t] - cs.norm_2(crowd_vel))  # Match velocity

            # Maintain safe distance
            ## TODO: placeholder for now
            min_distance = 0
            cost_safe = cs.exp(self.d_safe - min_distance)

            # Smooth control
            cost_smooth = 0
            if t > 0:
                cost_smooth = cs.sumsqr(u_var[:, t] - u_var[:, t - 1])

            cost += self.w_goal * cost_goal + self.w_follow * cost_follow + self.w_safe * cost_safe + self.w_smooth * cost_smooth
        
        opti.minimize(cost)
        
        # Constraints
        for t in range(self.mpc_horizon):
            opti.subject_to(x_var[:, t + 1] == mpc_utils.dynamics(x_var[:, t], u_var[:, t], self.dt))
            opti.subject_to(opti.bounded(-self.max_rev_speed, x_var[3, t], self.max_speed))
            opti.subject_to(opti.bounded(-self.max_rot, u_var[1, t], self.max_rot))
            opti.subject_to(opti.bounded(-self.max_l_dcc, u_var[0, t], self.max_l_acc))

        # Initial state constraint
        opti.subject_to(x_var[:, 0] == x_init)
        
        # Solver setup
        opts = {
            "ipopt.print_level": ipopt_print_level, 
            "print_time": False,
            "max_iter": 500,}
        opti.solver("ipopt", opts)
        
        # Store solver attributes
        self.opti = opti
        self.opti_dict = {
            "opti": opti,
            "x_var": x_var,
            "u_var": u_var,
            "x_init": x_init,
            "target": target,
        }

    def select_action(self, current_state, target, obs_data, pred_group_info):
        """Solves the MPC optimization and returns the next control action."""
        opti = self.opti

        # Set current state
        opti.set_value(self.opti_dict["x_init"], current_state)
        opti.set_value(self.opti_dict["target"], target)

        # Set human trajectory and velocity
        opti.set_value(self.opti_dict["obs_data"], obs_data)
        opti.set_value(self.opti_dict["pred_group_info"], pred_group_info)

        # Solve the MPC problem
        sol = opti.solve()

        # Extract first control action
        u_opt = sol.value(self.opti_dict["u_var"][:, 0])

        return u_opt, sol.value(self.opti_dict["x_var"])
    


