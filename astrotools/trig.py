import math
import numpy as np

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
