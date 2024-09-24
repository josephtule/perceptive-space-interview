import math
import numpy as np
from .trig import EAtoDCM

def COEtoRV(a, e, inc, raan, aop, ta, mu, degorrad="degrees", tol=1e-12):
    """
    Converts classical orbital elements to an inertial state vector. Units will depend on the 
    semimajor axis and the gravitational parameter. Angle units are defaulted to degrees.

    Parameters:
        a (float): Semimajor axis (km or m)
        e (float): Eccentricity
        inc (float): Inclination (degrees or radians)
        raan (float): Right ascension of ascending node (degrees or radians)
        aop (float): Argument of perigee (degrees or radians)
        ta (float): True anomaly (degrees or radians)
        mu (float): Gravitational constant for central body (km^3/s^2 or m^3/s^2)
        degorrad (str): String indicating if the angles are in degrees or radians (default "degrees")

    Returns:
        np.ndarray: Inertial state vector [r_x, r_y, r_z, v_x, v_y, v_z] in km or m.
    """

    if degorrad.lower() in ["deg","degrees","degree","degs"]:
        inc = np.deg2rad(inc)
        raan = np.deg2rad(raan)
        aop = np.deg2rad(aop)
        ta = np.deg2rad(ta)

    # determine orbit type and anomaly
    if e < tol and inc < tol:  # Circular equatorial
        anom = ta + raan + aop
        aop = 0
        raan = 0
    elif e < tol and inc > tol:  # Circular inclined
        anom = ta + aop
        aop = 0
    elif e > tol and inc < tol:  # Elliptic equatorial
        anom = raan + aop
        raan = 0
    else:
        anom = ta

    # Orbit parameter
    if e != 1:
        p = a * (1 - e ** 2)
    else:
        p = np.nan

    # Calculating state in the orbital plane relative to periapsis
    r_orbit = np.array([
        p * np.cos(anom) / (1 + e * np.cos(anom)),
        p * np.sin(anom) / (1 + e * np.cos(anom)),
        0
    ])

    v_orbit = np.array([
        -np.sqrt(mu / p) * np.sin(anom),
        np.sqrt(mu / p) * (e + np.cos(anom)),
        0
    ])

    # Rotation matrix from the perifocal frame to the inertial frame
    R, _ = EAtoDCM(np.array([-aop, -inc, -raan]), [3, 1, 3])

    # Inertial position and velocity
    r_inertial = np.dot(R, r_orbit)
    v_inertial = np.dot(R, v_orbit)

    # Return the inertial state vector 
    return np.concatenate((r_inertial, v_inertial))