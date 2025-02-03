import pandas as pd
import numpy as np

class DataLoader:
    
    def __init__(self, dataset):
        self.obs_delta = 2
        self.all_data = None
        
        if dataset == "atc":
            atc_file = "atc_1s_ds/1024.csv"
            self.all_data = self.load_atc_data(atc_file)
        
    def load_atc_data(self, atc_file):
        atc_data = pd.read_csv(atc_file, header=None, names=["time", "person_id", "x", "y", "velocity", "motion_angle"])
        atc_data['motion_angle'] = np.mod(atc_data['motion_angle'], 2 * np.pi)
        
        return atc_data
    
    def get_obs_data(self, obs_loc, obs_time, obs_r):
        if self.all_data is None:
            raise ValueError("Data not loaded. Please load the data using `load_atc_data`.")

        robot_x, robot_y = obs_loc

        time_filtered = self.all_data[
            (self.all_data["time"] >= obs_time) &
            (self.all_data["time"] <= obs_time + self.obs_delta)
        ].copy()

        time_filtered["distance"] = np.sqrt(
            (time_filtered["x"] - robot_x) ** 2 + (time_filtered["y"] - robot_y) ** 2
        )

        # Filter data by distance
        obs_data = time_filtered[time_filtered["distance"] <= obs_r]

        # Drop the distance column if not needed
        obs_data = obs_data.drop(columns=["distance"])
        
        # reset index
        obs_data.reset_index(drop=True, inplace=True)

        return obs_data
    
    def get_obs_data_by_time(self, obs_time):
        if self.all_data is None:
            raise ValueError("Data not loaded. Please load the data using `load_atc_data`.")

        obs_data = self.all_data[
            (self.all_data["time"] >= obs_time) &
            (self.all_data["time"] <= obs_time + self.obs_delta)
        ].copy()

        return obs_data

# atc_data_loader = DataLoader("atc")
# obs_time = 1351054538.983
# obs_loc = (40, -20)
# obs_r = 10
# obs_data = atc_data_loader.get_obs_data(obs_loc, obs_time, obs_r)
# obs_time_data = atc_data_loader.get_obs_data_by_time(obs_time)

# import plot_funcs as pf

# # pf.plot_atc_obs(obs_data, obs_time, obs_loc)
# pf.plot_atc_obs(obs_data, obs_time, obs_loc)


