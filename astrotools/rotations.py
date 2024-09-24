import math
import numpy as np
import pandas as pd
from .date_utils import *


def EAtoDCM(angles, seq, degorrad="radians"):
    """
    Computes a Direction Cosine Matrix (DCM) from Euler angles.

    Parameters:
        angles (np.ndarray): Array of three angles in order (3x1) <double>
        seq (np.ndarray): Array of three integers specifying the Euler angle sequence (3x1) <integer>
        degorrad (str): String flag for whether angles are in "degrees" or "radians" (default: "radians")

    Returns:
        rot (np.ndarray): The final multiplied 3x3 rotation matrix <double>
        Rout (list): A list of 3x3 individual rotation matrices <double>
    """

    # Convert to radians if the angles are in degrees
    if degorrad.lower() in ["deg", "degrees", "degree", "degs"]:
        angles = np.deg2rad(angles)

    # Define the rotation matrices
    def R1(t):
        return np.array([
            [1, 0, 0],
            [0, np.cos(t), np.sin(t)],
            [0, -np.sin(t), np.cos(t)]
        ])

    def R2(t):
        return np.array([
            [np.cos(t), 0, -np.sin(t)],
            [0, 1, 0],
            [np.sin(t), 0, np.cos(t)]
        ])

    def R3(t):
        return np.array([
            [np.cos(t), np.sin(t), 0],
            [-np.sin(t), np.cos(t), 0],
            [0, 0, 1]
        ])

    # Mapping to rotation matrices based on axis
    R = {1: R1, 2: R2, 3: R3}

    # Initialize output list and identity matrix for the final rotation matrix
    Rout = [None] * 3
    rot = np.eye(3)

    # Multiply the rotation matrices in reverse order of the sequence
    for i in range(2, -1, -1):
        rot = np.dot(rot, R[seq[i]](angles[i]))
        Rout[i] = R[seq[i]](angles[i])

    return rot, Rout


def rot(angle, axis, degorrad="radians"):
    """
    Computes a rotation matrix about a single axis.

    Parameters:
        angle (float): Rotation angle
        axis (int): Rotation axis (1 for x, 2 for y, 3 for z)
        degorrad (str): Flag for whether the angle is in "degrees" or "radians" (default: "radians")

    Returns:
        np.ndarray: The 3x3 rotation matrix
    """

    # Convert to radians if the angle is in degrees
    if degorrad.lower() in ["deg", "degrees", "degree", "degs"]:
        angle = np.deg2rad(angle)

    # Define the rotation matrix based on the axis
    if axis == 1:
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle), np.sin(angle)],
            [0, -np.sin(angle), np.cos(angle)]
        ])
    elif axis == 2:
        R = np.array([
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 3:
        R = np.array([
            [np.cos(angle), np.sin(angle), 0],
            [-np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 1, 2, or 3")

    return R


def DMStoDeg(degrees, minutes, seconds):
    """
    Converts degrees, minutes, and seconds to decimal degrees.
    """

    degs = np.abs(degrees) + np.abs(minutes) / 60 + np.abs(seconds) / 3600

    if degrees < 0.0:
        degs = -degs

    return degs


def RotNutation(JD, inputtime="TT", precision=106):
    """
    Computes the nutation rotation matrix using either a first-order or a higher-order approximation.
    
    Parameters:
        JD (float): Julian Date
        inputtime (str): Input time scale (default is "TT")
        precision (int or str): Number of terms for nutation calculation (default is 106)
    
    Returns:
        np.ndarray: The nutation rotation matrix (3x3)
    """
    # Import IAU1980 nutation parameters
    IAU1980 = np.array([[ 0.0,  0.0,  0.0,  0.0,  1.0, -6798.4, -171996.0,    -174.2,   92025.0,       8.9],     
         [ 0.0,  0.0,  2.0, -2.0,  2.0,   182.6,  -13187.0,      -1.6,    5736.0,      -3.1],     
         [ 0.0,  0.0,  2.0,  0.0,  2.0,    13.7,   -2274.0,      -0.2,     977.0,      -0.5],     
         [ 0.0,  0.0,  0.0,  0.0,  2.0, -3399.2,    2062.0,       0.2,    -895.0,       0.5],     
         [ 0.0, -1.0,  0.0,  0.0,  0.0,  -365.3,   -1426.0,       3.4,      54.0,      -0.1],     
         [ 1.0,  0.0,  0.0,  0.0,  0.0,    27.6,     712.0,       0.1,      -7.0,       0.0],     
         [ 0.0,  1.0,  2.0, -2.0,  2.0,   121.7,    -517.0,       1.2,     224.0,      -0.6],     
         [ 0.0,  0.0,  2.0,  0.0,  1.0,    13.6,    -386.0,      -0.4,     200.0,       0.0],     
         [ 1.0,  0.0,  2.0,  0.0,  2.0,     9.1,    -301.0,       0.0,     129.0,      -0.1],     
         [ 0.0, -1.0,  2.0, -2.0,  2.0,   365.2,     217.0,      -0.5,     -95.0,       0.3],     
         [-1.0,  0.0,  0.0,  2.0,  0.0,    31.8,     158.0,       0.0,      -1.0,       0.0],     
         [ 0.0,  0.0,  2.0, -2.0,  1.0,   177.8,     129.0,       0.1,     -70.0,       0.0],     
         [-1.0,  0.0,  2.0,  0.0,  2.0,    27.1,     123.0,       0.0,     -53.0,       0.0],     
         [ 1.0,  0.0,  0.0,  0.0,  1.0,    27.7,      63.0,       0.1,     -33.0,       0.0],     
         [ 0.0,  0.0,  0.0,  2.0,  0.0,    14.8,      63.0,       0.0,      -2.0,       0.0],     
         [-1.0,  0.0,  2.0,  2.0,  2.0,     9.6,     -59.0,       0.0,      26.0,       0.0],     
         [-1.0,  0.0,  0.0,  0.0,  1.0,   -27.4,     -58.0,      -0.1,      32.0,       0.0],     
         [ 1.0,  0.0,  2.0,  0.0,  1.0,     9.1,     -51.0,       0.0,      27.0,       0.0],     
         [-2.0,  0.0,  0.0,  2.0,  0.0,  -205.9,     -48.0,       0.0,       1.0,       0.0],     
         [-2.0,  0.0,  2.0,  0.0,  1.0,  1305.5,      46.0,       0.0,     -24.0,       0.0],     
         [ 0.0,  0.0,  2.0,  2.0,  2.0,     7.1,     -38.0,       0.0,      16.0,       0.0],     
         [ 2.0,  0.0,  2.0,  0.0,  2.0,     6.9,     -31.0,       0.0,      13.0,       0.0],     
         [ 2.0,  0.0,  0.0,  0.0,  0.0,    13.8,      29.0,       0.0,      -1.0,       0.0],     
         [ 1.0,  0.0,  2.0, -2.0,  2.0,    23.9,      29.0,       0.0,     -12.0,       0.0],     
         [ 0.0,  0.0,  2.0,  0.0,  0.0,    13.6,      26.0,       0.0,      -1.0,       0.0],     
         [ 0.0,  0.0,  2.0, -2.0,  0.0,   173.3,     -22.0,       0.0,       0.0,       0.0],     
         [-1.0,  0.0,  2.0,  0.0,  1.0,    27.0,      21.0,       0.0,     -10.0,       0.0],     
         [ 0.0,  2.0,  0.0,  0.0,  0.0,   182.6,      17.0,      -0.1,       0.0,       0.0],     
         [ 0.0,  2.0,  2.0, -2.0,  2.0,    91.3,     -16.0,       0.1,       7.0,       0.0],     
         [-1.0,  0.0,  0.0,  2.0,  1.0,    32.0,      16.0,       0.0,      -8.0,       0.0],     
         [ 0.0,  1.0,  0.0,  0.0,  1.0,   386.0,     -15.0,       0.0,       9.0,       0.0],     
         [ 1.0,  0.0,  0.0, -2.0,  1.0,   -31.7,     -13.0,       0.0,       7.0,       0.0],     
         [ 0.0, -1.0,  0.0,  0.0,  1.0,  -346.6,     -12.0,       0.0,       6.0,       0.0],     
         [ 2.0,  0.0, -2.0,  0.0,  0.0, -1095.2,      11.0,       0.0,       0.0,       0.0],     
         [-1.0,  0.0,  2.0,  2.0,  1.0,     9.5,     -10.0,       0.0,       5.0,       0.0],     
         [ 1.0,  0.0,  2.0,  2.0,  2.0,     5.6,      -8.0,       0.0,       3.0,       0.0],     
         [ 0.0, -1.0,  2.0,  0.0,  2.0,    14.2,      -7.0,       0.0,       3.0,       0.0],     
         [ 0.0,  0.0,  2.0,  2.0,  1.0,     7.1,      -7.0,       0.0,       3.0,       0.0],     
         [ 1.0,  1.0,  0.0, -2.0,  0.0,   -34.8,      -7.0,       0.0,       0.0,       0.0],     
         [ 0.0,  1.0,  2.0,  0.0,  2.0,    13.2,       7.0,       0.0,      -3.0,       0.0],     
         [-2.0,  0.0,  0.0,  2.0,  1.0,  -199.8,      -6.0,       0.0,       3.0,       0.0],     
         [ 0.0,  0.0,  0.0,  2.0,  1.0,    14.8,      -6.0,       0.0,       3.0,       0.0],     
         [ 2.0,  0.0,  2.0, -2.0,  2.0,    12.8,       6.0,       0.0,      -3.0,       0.0],     
         [ 1.0,  0.0,  0.0,  2.0,  0.0,     9.6,       6.0,       0.0,       0.0,       0.0],     
         [ 1.0,  0.0,  2.0, -2.0,  1.0,    23.9,       6.0,       0.0,      -3.0,       0.0],     
         [ 0.0,  0.0,  0.0, -2.0,  1.0,   -14.7,      -5.0,       0.0,       3.0,       0.0],     
         [ 0.0, -1.0,  2.0, -2.0,  1.0,   346.6,      -5.0,       0.0,       3.0,       0.0],     
         [ 2.0,  0.0,  2.0,  0.0,  1.0,     6.9,      -5.0,       0.0,       3.0,       0.0],     
         [ 1.0, -1.0,  0.0,  0.0,  0.0,    29.8,       5.0,       0.0,       0.0,       0.0],     
         [ 1.0,  0.0,  0.0, -1.0,  0.0,   411.8,      -4.0,       0.0,       0.0,       0.0],     
         [ 0.0,  0.0,  0.0,  1.0,  0.0,    29.5,      -4.0,       0.0,       0.0,       0.0],     
         [ 0.0,  1.0,  0.0, -2.0,  0.0,   -15.4,      -4.0,       0.0,       0.0,       0.0],     
         [ 1.0,  0.0, -2.0,  0.0,  0.0,   -26.9,       4.0,       0.0,       0.0,       0.0],     
         [ 2.0,  0.0,  0.0, -2.0,  1.0,   212.3,       4.0,       0.0,      -2.0,       0.0],     
         [ 0.0,  1.0,  2.0, -2.0,  1.0,   119.6,       4.0,       0.0,      -2.0,       0.0],     
         [ 1.0,  1.0,  0.0,  0.0,  0.0,    25.6,      -3.0,       0.0,       0.0,       0.0],     
         [ 1.0, -1.0,  0.0, -1.0,  0.0, -3232.9,      -3.0,       0.0,       0.0,       0.0],     
         [-1.0, -1.0,  2.0,  2.0,  2.0,     9.8,      -3.0,       0.0,       1.0,       0.0],     
         [ 0.0, -1.0,  2.0,  2.0,  2.0,     7.2,      -3.0,       0.0,       1.0,       0.0],     
         [ 1.0, -1.0,  2.0,  0.0,  2.0,     9.4,      -3.0,       0.0,       1.0,       0.0],     
         [ 3.0,  0.0,  2.0,  0.0,  2.0,     5.5,      -3.0,       0.0,       1.0,       0.0],     
         [-2.0,  0.0,  2.0,  0.0,  2.0,  1615.7,      -3.0,       0.0,       1.0,       0.0],     
         [ 1.0,  0.0,  2.0,  0.0,  0.0,     9.1,       3.0,       0.0,       0.0,       0.0],     
         [-1.0,  0.0,  2.0,  4.0,  2.0,     5.8,      -2.0,       0.0,       1.0,       0.0],     
         [ 1.0,  0.0,  0.0,  0.0,  2.0,    27.8,      -2.0,       0.0,       1.0,       0.0],     
         [-1.0,  0.0,  2.0, -2.0,  1.0,   -32.6,      -2.0,       0.0,       1.0,       0.0],     
         [ 0.0, -2.0,  2.0, -2.0,  1.0,  6786.3,      -2.0,       0.0,       1.0,       0.0],     
         [-2.0,  0.0,  0.0,  0.0,  1.0,   -13.7,      -2.0,       0.0,       1.0,       0.0],     
         [ 2.0,  0.0,  0.0,  0.0,  1.0,    13.8,       2.0,       0.0,      -1.0,       0.0],     
         [ 3.0,  0.0,  0.0,  0.0,  0.0,     9.2,       2.0,       0.0,       0.0,       0.0],     
         [ 1.0,  1.0,  2.0,  0.0,  2.0,     8.9,       2.0,       0.0,      -1.0,       0.0],     
         [ 0.0,  0.0,  2.0,  1.0,  2.0,     9.3,       2.0,       0.0,      -1.0,       0.0],     
         [ 1.0,  0.0,  0.0,  2.0,  1.0,     9.6,      -1.0,       0.0,       0.0,       0.0],     
         [ 1.0,  0.0,  2.0,  2.0,  1.0,     5.6,      -1.0,       0.0,       1.0,       0.0],     
         [ 1.0,  1.0,  0.0, -2.0,  1.0,   -34.7,      -1.0,       0.0,       0.0,       0.0],     
         [ 0.0,  1.0,  0.0,  2.0,  0.0,    14.2,      -1.0,       0.0,       0.0,       0.0],     
         [ 0.0,  1.0,  2.0, -2.0,  0.0,   117.5,      -1.0,       0.0,       0.0,       0.0],     
         [ 0.0,  1.0, -2.0,  2.0,  0.0,  -329.8,      -1.0,       0.0,       0.0,       0.0],     
         [ 1.0,  0.0, -2.0,  2.0,  0.0,    23.8,      -1.0,       0.0,       0.0,       0.0],     
         [ 1.0,  0.0, -2.0, -2.0,  0.0,    -9.5,      -1.0,       0.0,       0.0,       0.0],     
         [ 1.0,  0.0,  2.0, -2.0,  0.0,    32.8,      -1.0,       0.0,       0.0,       0.0],     
         [ 1.0,  0.0,  0.0, -4.0,  0.0,   -10.1,      -1.0,       0.0,       0.0,       0.0],     
         [ 2.0,  0.0,  0.0, -4.0,  0.0,   -15.9,      -1.0,       0.0,       0.0,       0.0],     
         [ 0.0,  0.0,  2.0,  4.0,  2.0,     4.8,      -1.0,       0.0,       0.0,       0.0],     
         [ 0.0,  0.0,  2.0, -1.0,  2.0,    25.4,      -1.0,       0.0,       0.0,       0.0],     
         [-2.0,  0.0,  2.0,  4.0,  2.0,     7.3,      -1.0,       0.0,       1.0,       0.0],     
         [ 2.0,  0.0,  2.0,  2.0,  2.0,     4.7,      -1.0,       0.0,       0.0,       0.0],     
         [ 0.0, -1.0,  2.0,  0.0,  1.0,    14.2,      -1.0,       0.0,       0.0,       0.0],     
         [ 0.0,  0.0, -2.0,  0.0,  1.0,   -13.6,      -1.0,       0.0,       0.0,       0.0],     
         [ 0.0,  0.0,  4.0, -2.0,  2.0,    12.7,       1.0,       0.0,       0.0,       0.0],     
         [ 0.0,  1.0,  0.0,  0.0,  2.0,   409.2,       1.0,       0.0,       0.0,       0.0],     
         [ 1.0,  1.0,  2.0, -2.0,  2.0,    22.5,       1.0,       0.0,      -1.0,       0.0],     
         [ 3.0,  0.0,  2.0, -2.0,  2.0,     8.7,       1.0,       0.0,       0.0,       0.0],     
         [-2.0,  0.0,  2.0,  2.0,  2.0,    14.6,       1.0,       0.0,      -1.0,       0.0],     
         [-1.0,  0.0,  0.0,  0.0,  2.0,   -27.3,       1.0,       0.0,      -1.0,       0.0],     
         [ 0.0,  0.0, -2.0,  2.0,  1.0,  -169.0,       1.0,       0.0,       0.0,       0.0],     
         [ 0.0,  1.0,  2.0,  0.0,  1.0,    13.1,       1.0,       0.0,       0.0,       0.0],     
         [-1.0,  0.0,  4.0,  0.0,  2.0,     9.1,       1.0,       0.0,       0.0,       0.0],     
         [ 2.0,  1.0,  0.0, -2.0,  0.0,   131.7,       1.0,       0.0,       0.0,       0.0],     
         [ 2.0,  0.0,  0.0,  2.0,  0.0,     7.1,       1.0,       0.0,       0.0,       0.0],     
         [ 2.0,  0.0,  2.0, -2.0,  1.0,    12.8,       1.0,       0.0,      -1.0,       0.0],     
         [ 2.0,  0.0, -2.0,  0.0,  1.0,  -943.2,       1.0,       0.0,       0.0,       0.0],     
         [ 1.0, -1.0,  0.0, -2.0,  0.0,   -29.3,       1.0,       0.0,       0.0,       0.0],     
         [-1.0,  0.0,  0.0,  1.0,  1.0,  -388.3,       1.0,       0.0,       0.0,       0.0],     
         [-1.0, -1.0,  0.0,  2.0,  1.0,    35.0,       1.0,       0.0,       0.0,       0.0],     
         [ 0.0,  1.0,  0.0,  1.0,  0.0,    27.3,       1.0,       0.0,       0.0,       0.0]])

    # Time conversion
    T = (JD - 2451545.0) / 36525.0
    T = timeconverter(T, inputtime, "TT")
    T /= (100 * 365.25 * 24 * 60 * 60)  # Convert back to centuries

    epsilon = 23.43929111 * 3600 - 46.8150 * T - 0.00059 * T**2 + 0.001813 * T**3

    if precision in ["approx", "approximate"]:
        l = 357.525 + 35999 * T  # mean anomaly of the sun [deg]
        F = 93.273 + 483202.019 * T  # mean distance b/w nodes of moon [deg]
        D = 297.850 + 445267.111 * T  # mean distance b/w sun and moon [deg]
        Omega = 125.045 - 1934.136 * T  # mean longitude of moon [deg]

        delta_Psi = -17.200 * np.sin(np.deg2rad(Omega)) \
                    + 0.202 * np.sin(np.deg2rad(2 * Omega)) \
                    - 1.319 * np.sin(np.deg2rad(2 * (F - D + Omega))) \
                    + 0.143 * np.sin(np.deg2rad(l)) \
                    - 0.227 * np.sin(np.deg2rad(2 * (F + Omega)))
        delta_epsilon = 9.203 * np.cos(np.deg2rad(Omega)) \
                        - 0.090 * np.cos(np.deg2rad(2 * Omega)) \
                        - 0.547 * np.cos(np.deg2rad(2 * (F - D + Omega))) \
                        + 0.098 * np.cos(np.deg2rad(2 * (F + Omega)))
    else:
        if isinstance(precision, str):
            precision = 106

        l = DMStoDeg(134, 57, 46.733) + DMStoDeg(477198, 52, 2.633) * T
        l_prime = DMStoDeg(357, 31, 39.804) + DMStoDeg(35999, 3, 1.244) * T
        F = DMStoDeg(93, 16, 18.877) + DMStoDeg(483202, 1, 3.137) * T
        D = DMStoDeg(297, 51, 1.307) + DMStoDeg(445267, 6, 41.328) * T
        Omega = DMStoDeg(125, 2, 40.280) - DMStoDeg(1934, 8, 10.539) * T

        delta_Psi = 0
        delta_epsilon = 0
        for i in range(precision):
            phi = IAU1980[i, 0] * l + IAU1980[i, 1] * l_prime + IAU1980[i, 2] * F + IAU1980[i, 3] * D + IAU1980[i, 4] * Omega
            delta_Psi += (IAU1980[i, 6] + IAU1980[i, 7] * T) * np.sin(np.deg2rad(phi)) / 10000
            delta_epsilon += (IAU1980[i, 8] + IAU1980[i, 9] * T) * np.cos(np.deg2rad(phi)) / 10000

    # Convert rotation angles from seconds to degrees
    epsilon /= 3600
    delta_Psi /= 3600
    delta_epsilon /= 3600

    # Compute the nutation rotation matrix
    N = np.dot(rot(-epsilon - delta_epsilon, 1, "degrees"), 
               np.dot(rot(-delta_Psi, 3, "degrees"), 
                      rot(epsilon, 1, "degrees")))
    
    return N


def RotPolarMotion(JD, EOP2=None, precision="full"):
    """
    Computes the polar motion rotation matrix for a given Julian Date (JD)
    using Earth Orientation Parameters (EOP) data.
    
    Parameters:
        JD (float): Julian Date
        EOP2 (pd.DataFrame): Earth Orientation Parameters data as a pandas DataFrame.
        precision (str): Either "full" for full precision or "approx" for small-angle approximation.
                         Defaults to "full".
    
    Returns:
        np.array: 3x3 polar motion rotation matrix (PM)
    """
    
    if EOP2 is None:
        EOP2 = parseEOPFile("EOP2long.txt")

    # Convert JD to MJD
    MJD = JDtoMJD(JD)
    
    # Find the closest MJD in the EOP2 data
    idx = (EOP2['MJD'] - MJD).abs().idxmin()  # Index of closest MJD value
    
    # Get the closest PMx and PMy values, convert from milli-arcseconds to degrees
    xp = EOP2.loc[idx, 'PMx'] / 3600 / 1000
    yp = EOP2.loc[idx, 'PMy'] / 3600 / 1000
    
    # Generate the polar motion rotation matrix
    if precision == "full":
        PM = np.dot(rot(-xp, 2, "degrees"), rot(-yp, 1, "degrees"))
    else:
        # Small-angle approximation
        xp = np.deg2rad(xp)
        yp = np.deg2rad(yp)
        PM = np.array([[1, 0, xp],
                       [0, 1, -yp],
                       [-xp, yp, 1]])
    
    return PM


def parseEOPFile(filename):
    """
    Parses an Earth Orientation Parameters (EOP) file and extracts the data 
    into a pandas DataFrame.
    
    Parameters:
        filename (str): Path to the EOP file
    
    Returns:
        data (pd.DataFrame): Parsed EOP data with columns:
            ['MJD', 'PMx', 'PMy', 'TAI_UT1', 'PMxSig', 'PMySig', 'UTSig', 
             'UTPM_Corr_1', 'UTPM_Corr_2', 'UTPM_Corr_3', 'DX', 'DY', 
             'DXSig', 'DYSig', 'Corr']
    """
    
    # Initialize empty lists to hold the data
    MJD = []
    PMx = []
    PMy = []
    TAI_UT1 = []
    PMxSig = []
    PMySig = []
    UTSig = []
    UTPM_Corr_1 = []
    UTPM_Corr_2 = []
    UTPM_Corr_3 = []
    DX = []
    DY = []
    DXSig = []
    DYSig = []
    Corr = []
    
    # Open the file and read line by line
    with open(filename, 'r') as file:
        for line in file:
            # Skip lines starting with '$' or 'EOP2=' or empty lines
            if line.startswith(' $') or line.startswith(' EOP2=') or line.strip() == '':
                continue
            
            # Only keep part before the '$' symbol
            line = line.split('$')[0].strip()

            # Skip empty lines after splitting
            if not line:
                continue
            
            # Parse the line and extract the data
            data = [float(val) for val in line.split(',') if val.strip()]
            
            # Unpack the extracted data
            mjd, pmx, pmy, tai_ut1, pmx_sig, pmy_sig, ut_sig, utpm_corr1, utpm_corr2, utpm_corr3, dx, dy, dx_sig, dy_sig, corr = data
            
            # Append data to respective lists
            MJD.append(mjd)
            PMx.append(pmx)
            PMy.append(pmy)
            TAI_UT1.append(tai_ut1)
            PMxSig.append(pmx_sig)
            PMySig.append(pmy_sig)
            UTSig.append(ut_sig)
            UTPM_Corr_1.append(utpm_corr1)
            UTPM_Corr_2.append(utpm_corr2)
            UTPM_Corr_3.append(utpm_corr3)
            DX.append(dx)
            DY.append(dy)
            DXSig.append(dx_sig)
            DYSig.append(dy_sig)
            Corr.append(corr)
    
    # Create a pandas DataFrame with the parsed data
    data = pd.DataFrame({
        'MJD': MJD,
        'PMx': PMx,
        'PMy': PMy,
        'TAI_UT1': TAI_UT1,
        'PMxSig': PMxSig,
        'PMySig': PMySig,
        'UTSig': UTSig,
        'UTPM_Corr_1': UTPM_Corr_1,
        'UTPM_Corr_2': UTPM_Corr_2,
        'UTPM_Corr_3': UTPM_Corr_3,
        'DX': DX,
        'DY': DY,
        'DXSig': DXSig,
        'DYSig': DYSig,
        'Corr': Corr
    })
    
    return data


def RotPrecession(JD, inputtime="TT"):
    """
    Computes the precession rotation matrix for a given Julian Date (JD)
    using the IAU 1976 precession model.
    
    Parameters:
        JD (float): Julian Date
        inputtime (str): Time scale for input JD. Options are "TT", 
                         "UTC", "UT1", or "TAI". Defaults to "TT".
    
    Returns:
        np.array: 3x3 precession rotation matrix
    """
      
    # Compute T in seconds to adjust for different times
    T = (JD - 2451545.0)/36525.0 * (100. * 365.25 * 24. * 60. * 60.)

    # Convert to terrestrial time
    T = timeconverter(T, inputtime, "TT")
    T /= (100. * 365.25 * 24. * 60. * 60.) # convert back to centuries

    # Compute precession angles
    zeta = 2306.2181 * T + 0.30188 * T**2 + 0.017998 * T**3
    theta = 2004.3109 * T - 0.42665 * T**2 - 0.041833 * T**3
    z = zeta + 0.79280 * T**2 + 0.000205 * T**3 

    # Convert from seconds to degrees
    zeta /= 3600
    theta /= 3600
    z /= 3600

    # Compute rotation matrix
    return np.dot(rot(-z, 3, "degrees"), np.dot(rot(theta, 2, "degrees"), rot(-zeta, 3, "degrees")))


def GEODtoECEF(lat, lon, h, body, degorrad="degrees"):
    """
    Converts geodetic coordinates to ECEF (Earth-Centered Earth-Fixed) coordinates.
    
    Parameters:
        lat (float or np.array): Geodetic latitude
        lon (float or np.array): Longitude
        h (float or np.array): Height above ellipsoid
        body (dict): Dictionary containing body properties (e.g., flattening, semimajor axis)
        degorrad (str): Flag indicating degrees or radians (default is "degrees")
        
    Returns:
        np.array: Position vector in ECEF (3 x n)
    """
    # Convert to radians if input is in degrees
    if degorrad.lower() in ["deg", "degrees", "degree", "degs"]:
        lat = np.radians(lat)
        lon = np.radians(lon)
    
    # Extract body properties
    f = body.Flattening
    a = body.SemimajorAxis
    
    # Calculate eccentricity squared
    e2 = 2 * f - f**2
    
    # Calculate the radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    
    # Compute the ECEF coordinates
    r_ecef = np.array([
        (N + h) * np.cos(lat) * np.cos(lon),  # X
        (N + h) * np.cos(lat) * np.sin(lon),  # Y
        (N * (1 - e2) + h) * np.sin(lat)      # Z
    ])
    
    return r_ecef


