import logging
import time
from copy import deepcopy
import os

import numpy as np
import pandas as pd
import casadi as cs

from matplotlib import pyplot as plt
import pickle

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.stats import vonmises, norm

from data_loader import DataLoader
from utils import mpc_utils


def compute_distance_matrix(features):
    """
    Compute a pairwise distance matrix for DBSCAN using standardized distances.
    Features: (x, y, velocity, motion_angle in radians)
    """
    num_samples = len(features)
    distance_matrix = np.zeros((num_samples, num_samples))

    location_distances = np.zeros((num_samples, num_samples))
    velocity_distances = np.zeros((num_samples, num_samples))
    angle_distances = np.zeros((num_samples, num_samples))

    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            u, v = features[i], features[j]

            spatial_distance = np.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)  # (x, y)
            velocity_distance = abs(u[2] - v[2])  # Speed difference
            angle_distance = mpc_utils.circdiff(u[3], v[3])  # Motion angle difference

            location_distances[i, j] = spatial_distance
            velocity_distances[i, j] = velocity_distance
            angle_distances[i, j] = angle_distance

    # Reflect to full symmetric matrix
    location_distances += location_distances.T
    velocity_distances += velocity_distances.T
    angle_distances += angle_distances.T

    # Standardize each distance type (subtract mean, divide by std)
    location_distances = (location_distances - np.mean(location_distances)) / np.std(location_distances)
    velocity_distances = (velocity_distances - np.mean(velocity_distances)) / np.std(velocity_distances)
    angle_distances = (angle_distances - np.mean(angle_distances)) / np.std(angle_distances)

    # Shift distances so that the minimum value is zero
    location_distances -= np.min(location_distances)
    velocity_distances -= np.min(velocity_distances)
    angle_distances -= np.min(angle_distances)

    # Combine standardized distances
    distance_matrix = location_distances + velocity_distances + angle_distances

    return distance_matrix


def compute_group_centroid(group_data):
    centroid_x = group_data['x'].mean()
    centroid_y = group_data['y'].mean()
    
    return centroid_x, centroid_y


def compute_group_motion(group_data):
    motion_angle_data = group_data['motion_angle'].values
    mu_theta, kappa, _ = vonmises.fit(motion_angle_data)

    speed_data = group_data['velocity'].values
    mu_speed, sigma_speed = norm.fit(speed_data)
    
    ### compute the average speed and motion angle
    weights = np.ones_like(motion_angle_data)
    avg_motion_angle = mpc_utils.circmean(motion_angle_data, weights)
    avg_speed = group_data['velocity'].mean()
    
    # print("-------- check the average speed and motion angle --------")
    # print(mu_speed, sigma_speed, mu_theta, kappa)
    # print(avg_speed, avg_motion_angle)
    # print("----------------------------------------------------------")

    return avg_speed, avg_motion_angle


def compute_group_properties(obs_data):
    group_list = []

    for cluster_id in obs_data['group'].unique():
        if cluster_id == -1:  # Skip noise points
            continue

        group_data = obs_data[obs_data['group'] == cluster_id]
        centroid_x, centroid_y = compute_group_centroid(group_data)
        avg_speed, avg_motion_angle = compute_group_motion(group_data)

        group_list.append([cluster_id, centroid_x, centroid_y, avg_speed, avg_motion_angle])
    
    group_array = np.array(group_list)
    
    return group_array


class CrowdModel:
    def __init__(self, data_loader, config):
        if config is not None:
            self.configure(config)
        else:
            raise ValueError('Please provide a configuration file')
        
        self.data_loader = data_loader

    def configure(self, config):
        self.pred_horizon = config.getint('mpc_env', 'mpc_horizon')
        logging.info('[MPCEnv] Config {:} = {:}'.format('pred_horizon', self.pred_horizon))
        
        self.dt = config.getfloat('mpc_env', 'dt')
        logging.info('[MPCEnv] Config {:} = {:}'.format('dt', self.dt))
        
        self.obs_radius = config.getfloat('mpc_env', 'obs_radius')
        logging.info('[MPCEnv] Config {:} = {:}'.format('obs_radius', self.obs_radius))

    def get_group_status(self, obs_time, obs_loc):
        obs_data = self.data_loader.get_obs_data(obs_loc, obs_time, self.obs_radius)
        obs_time_data = self.data_loader.get_obs_data_by_time(obs_time)
        
        obs_data = self.cluster_humans(obs_data)
        # obs_data = self.cluster_humans_v2(obs_data)
        
        pred_group_info = self.group_property_prediction(obs_data)

        # Plot results
        self.plot_clusters(obs_data)
        self.plot_predictions(pred_group_info)

        return obs_data, pred_group_info
    
    def plot_clusters(self, obs_data):
        plt.figure(figsize=(8, 6))
        unique_labels = obs_data['group'].unique()

        for cluster_id in unique_labels:
            cluster_points = obs_data[obs_data['group'] == cluster_id]
            # print how many people in each cluster
            print("-----------------------------")
            print(f"Cluster {cluster_id} has {len(cluster_points)} people")
            plt.scatter(cluster_points['x'], cluster_points['y'], label=f'Cluster {cluster_id}')
            (u, v) = mpc_utils.pol2cart(cluster_points["velocity"], cluster_points["motion_angle"])
            plt.quiver(cluster_points['x'], cluster_points['y'], 
                       u, v,
                       angles='xy', scale_units='xy', scale=1, color='black', alpha=0.5)

        plt.legend()
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Pedestrian Clustering Results')
        plt.show()
        

    def plot_predictions(self, pred_group_info):
        ### pred_group_info: [time_step, cluster_id, new_x, new_y, avg_speed, avg_motion_angle] ###
        ### num_clusters, self.pred_horizon, 6
        
        plt.figure(figsize=(8, 6))
        num_clusters = pred_group_info.shape[0]
        
        for i in range(num_clusters):
            cluster_id = int(pred_group_info[i, 0, 1])
            cluster_trajectory = pred_group_info[i]
            
            plt.plot(cluster_trajectory[:, 2], cluster_trajectory[:, 3], marker='o', linestyle='--', label=f'Cluster {cluster_id}')
        
        # unique_clusters = np.unique(pred_group_info[:, 1])

        # for cluster_id in unique_clusters:
        #     cluster_predictions = pred_group_info[pred_group_info[:, 1] == cluster_id]

        #     plt.plot(cluster_predictions[:, 2], cluster_predictions[:, 3], marker='o', linestyle='--',
        #              label=f'Predicted Path (Cluster {int(cluster_id)})')

        plt.legend()
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Predicted Future Locations of Groups')
        plt.show()
        
    def cluster_humans(self, obs_data):
        feature_vectors = obs_data[['x', 'y', 'velocity', 'motion_angle']].values  
        distance_matrix = compute_distance_matrix(feature_vectors)
        dbscan = DBSCAN(eps=3, min_samples=2, metric='precomputed')  # eps is now easier to tune
        labels = dbscan.fit_predict(distance_matrix)
        obs_data['group'] = labels
        
        ###For debugging##########################
        # # print index of obs_data
        # print("Index of obs_data:")
        # print(obs_data.index)
        # print("for debugging start____________________")
        # print("Total number of motions: ", len(obs_data['motion_angle']))
        # green_point_index = obs_data[obs_data['group'] == -1].index[0]
        # print(f"Green Point Index: {green_point_index}")
        
        # # Step 2: Find a nearby blue point based on location
        # green_point_position = obs_data.loc[green_point_index, ['x', 'y']].values
        # ## print green point all features:
        # print("Green Point Features:")
        # print(obs_data.loc[green_point_index])
        
        # # Get all blue points (Cluster 0)
        # blue_points = obs_data[obs_data['group'] == 0]

        # # Compute Euclidean distances to all blue points
        # blue_point_positions = blue_points[['x', 'y']].values
        # distances_to_blue_points = np.linalg.norm(blue_point_positions - green_point_position, axis=1)

        # # Find the nearest blue point
        # nearest_blue_point_idx = blue_points.index[np.argmin(distances_to_blue_points)]
        # print(f"Nearest Blue Point Index: {nearest_blue_point_idx}")
        # # print nearest blue point all features:
        
        # print("Nearest Blue Point Features:")
        # print(obs_data.loc[nearest_blue_point_idx])
        # # Step 3: Extract distance matrix rows
        # green_point_distances = distance_matrix[green_point_index]
        # blue_point_distances = distance_matrix[nearest_blue_point_idx]
        # print("Distances from one blue point to others:")
        # print(blue_point_distances)
        # print("Distances from one green point to others:")
        # print(green_point_distances)
        ##########################################

        return obs_data
    
    # a simple version for computing distance matrix
    def cluster_humans_v2(self, obs_data):
        feature_vectors = obs_data[['x', 'y', 'velocity', 'motion_angle']].copy()
        feature_vectors["motion_angle_sin"] = np.sin(feature_vectors["motion_angle"])
        feature_vectors["motion_angle_cos"] = np.cos(feature_vectors["motion_angle"])
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_vectors)
        dbscan = DBSCAN(eps=1, min_samples=5)
        labels = dbscan.fit_predict(features_scaled)
        obs_data['group'] = labels
        
        return obs_data

    def group_property_prediction(self, obs_data):
        group_info = compute_group_properties(obs_data)
        num_clusters = group_info.shape[0]
        # pred_group_info = np.zeros((self.pred_horizon * num_clusters, 6))
        pred_group_info = np.zeros((num_clusters, self.pred_horizon, 6))                  
        
        # row_idx = 0
        
        for i, row in enumerate(group_info):
            cluster_id, centroid_x, centroid_y, avg_speed, avg_motion_angle = row
            for time_step in range(self.pred_horizon):
                centroid_x += avg_speed * np.cos(avg_motion_angle) * self.dt
                centroid_y += avg_speed * np.sin(avg_motion_angle) * self.dt
                pred_group_info[i, time_step] = [time_step + 1, cluster_id, centroid_x, centroid_y, avg_speed, avg_motion_angle]

        
        # for row in group_info:
        #     cluster_id, centroid_x, centroid_y, avg_speed, avg_motion_angle = row
        #     for time_step in range(1, self.pred_horizon + 1):
        #         centroid_x += avg_speed * np.cos(avg_motion_angle) * self.dt
        #         centroid_y += avg_speed * np.sin(avg_motion_angle) * self.dt
        #         pred_group_info[row_idx] = [time_step, cluster_id, centroid_x, centroid_y, avg_speed, avg_motion_angle]
        #         row_idx += 1

        return pred_group_info


# atc_data_loader = DataLoader("atc")
# config = mpc_utils.parse_config_file("configs/crowd_mpc.config")
# crowd_model = CrowdModel(atc_data_loader, config)
# # obs_time = 1351057157.972
# obs_time = 1351047458.0
# obs_loc = (15, -10)
# obs_data, pred_group_info = crowd_model.get_group_status(obs_time, obs_loc)
# # print(obs_data)
# # print(pred_group_info)