import numpy as np

def angular_diff(angle1, angle2):
    return np.abs(np.arctan2(np.sin(angle1 - angle2), np.cos(angle1- angle2)))

angle1 = np.deg2rad(-1)
angle2 = np.deg2rad(359)
print(np.rad2deg(angular_diff(angle1, angle2)))