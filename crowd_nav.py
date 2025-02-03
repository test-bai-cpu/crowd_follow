import numpy as np
import pandas as pd
from typing import Tuple
import time
import logging

from data_loader import DataLoader
from crowd_aware_MPC import CrowdAwareMPC
from crowd_model import CrowdModel
from utils import mpc_utils


class CrowdNav:
    def __init__(
        self,
        config,
        init_state : np.ndarray,
        target: Tuple[float, float],
        start_time: float,
        crowd_model: CrowdModel,
        mpc: CrowdAwareMPC
        ):
        if config is not None:
            self.configure(config)
        else:
            raise ValueError('Please provide a configuration file')
        self.init_state = init_state
        self.target = target
        self.start_time = start_time
        self.crowd_model = crowd_model
        self.mpc = mpc

    def configure(self, config):
        self.dt = config.getfloat('mpc_env', 'dt')
        logging.info('[MPCEnv] Config {:} = {:}'.format('dt', self.dt))
        
    def get_plan(self):
        trajectory = []
        current_state = self.init_state
        current_time = self.start_time
        for i in range(10):
            obs_data, pred_group_info = self.crowd_model.get_group_status(current_time, current_state[:2])
            start_time = time.time()
            control_action, predicted_trajectory = self.mpc.select_action(
                current_state, self.target, obs_data, pred_group_info
            )
            end_time = time.time()
            current_state = mpc_utils.dynamics(current_state, control_action, self.dt)
            trajectory.append(current_state[:2])
            
            current_time += self.dt
            
        return np.array(trajectory)

config = mpc_utils.parse_config_file("configs/crowd_mpc.config")
atc_data_loader = DataLoader("atc")
crowd_model = CrowdModel(atc_data_loader, config)
mpc = CrowdAwareMPC(config)

crowd_nav = CrowdNav(
    config,
    np.array([1.0, 1.0, 0.0, 0.5]),
    (10, 10),
    0,
    crowd_model,
    mpc
    )
crowd_nav.get_plan()