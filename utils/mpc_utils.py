import casadi as cs
import configparser

import numpy as np


def dynamics(state, control, dt):
    x, y, theta, v = state[0], state[1], state[2], state[3]
    a, omega = control[0], control[1]
    next_state = cs.vertcat(
        x + v * cs.cos(theta) * dt,
        y + v * cs.sin(theta) * dt,
        theta + omega * dt,
        v + a * dt
    )
    return next_state

def parse_config_file(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    
    return config

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def circdiff(circular_1, circular_2):
    res = np.arctan2(np.sin(circular_1-circular_2), np.cos(circular_1-circular_2))
    return abs(res)

def wrapTo2pi(circular_value):
    return np.round(np.mod(circular_value,2*np.pi), 3)

def _circfuncs_common(samples, high, low):
    # Ensure samples are array-like and size is not zero
    if samples.size == 0:
        return np.nan, np.asarray(np.nan), np.asarray(np.nan), None

    # Recast samples as radians that range between 0 and 2 pi and calculate
    # the sine and cosine
    sin_samp = np.sin((samples - low)*2.* np.pi / (high - low))
    cos_samp = np.cos((samples - low)*2.* np.pi / (high - low))

    return samples, sin_samp, cos_samp

def circmean(samples, weights, high=2*np.pi, low=0):
    samples = np.asarray(samples)
    weights = np.asarray(weights)
    samples, sin_samp, cos_samp = _circfuncs_common(samples, high, low)
    sin_sum = sum(sin_samp * weights)
    cos_sum = sum(cos_samp * weights)
    res = np.arctan2(sin_sum, cos_sum)
    res = res*(high - low)/2.0/np.pi + low
    return wrapTo2pi(res)