import math
import numpy as np
import pandas as pd
from .rotations import *
from .mathtools import *


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


def RVtoCOE(RV, mu, degorrad="degrees", tol=1e-7):
    """
    Converts an inertial state vector to classical orbital elements (COEs).

    Parameters:
        RV (np.ndarray): Inertial state vector [r_x, r_y, r_z, v_x, v_y, v_z] (km or m)
        mu (float): Gravitational constant of the central body (km^3/s^2 or m^3/s^2)
        degorrad (str): String indicating degrees or radians for angle outputs (default is "degrees")

    Returns:
        tuple: (a, e, inc, raan, aop, ta) 
               - a (float): Semimajor axis (km or m)
               - e (float): Eccentricity
               - inc (float): Inclination (degrees or radians)
               - raan (float): Right Ascension of the Ascending Node (degrees or radians)
               - aop (float): Argument of Perigee (degrees or radians)
               - ta (float): True Anomaly (degrees or radians)
    """

    # Extract position (r) and velocity (v) vectors
    r_inertial = RV[:3] # Position vector
    v_inertial = RV[3:] # Velocity vector
    r_mag = np.linalg.norm(r_inertial)  # Magnitude of position
    v_mag = np.linalg.norm(v_inertial)  # Magnitude of velocity

    # Specific angular momentum vector
    h_inertial = np.cross(r_inertial.T, v_inertial.T).T  # Cross product of r and v
    h_mag = np.linalg.norm(h_inertial)  # Magnitude of angular momentum

    # Inclination
    inc = np.arccos(h_inertial[2] / h_mag)

    # Node vector
    N_inertial = np.cross([0, 0, 1], h_inertial.T).T
    N_mag = np.linalg.norm(N_inertial)

    # RAAN (Right Ascension of Ascending Node)
    raan = np.arccos(N_inertial[0] / N_mag)
    if N_inertial[1] < 0:
        raan = 2 * np.pi - raan

    # Radial velocity component
    v_r = np.dot(r_inertial, v_inertial) / r_mag

    # Eccentricity vector
    e_inertial = (1 / mu) * ((v_mag**2 - mu / r_mag) * r_inertial - np.dot(r_inertial.T, v_inertial.T) * v_inertial)
    e = np.linalg.norm(e_inertial)

    # Argument of Perigee (AOP)
    aop = np.arccos(np.dot(N_inertial.T, e_inertial.T) / (N_mag * e))
    if e_inertial[2] < 0:
        aop = 2 * np.pi - aop

    # True Anomaly (TA)
    ta = np.real(np.arccos(np.dot(e_inertial.T, r_inertial.T) / (e * r_mag)))
    if v_r < 0:
        ta = 2 * np.pi - ta

    # Semimajor axis (a)
    specific_energy = (v_mag**2) / 2 - mu / r_mag
    a = -mu / (2 * specific_energy)

    # Handle angles close to 2π
    if abs(raan - 2 * np.pi) < tol:
        raan = 0
    if abs(aop - 2 * np.pi) < tol:
        aop = 0

    # Convert to degrees if required
    if degorrad.lower() in ["deg", "degrees", "degree", "degs"]:
        inc = np.rad2deg(inc)
        raan = np.rad2deg(raan)
        aop = np.rad2deg(aop)
        ta = np.rad2deg(ta)

    return a, e, inc, raan, aop, ta


def parseTLE(filename, mu, millenium):
    """
    Parses a TLE file to extract satellite information and compute orbital elements.
    
    Args:
        filename (str): Name of the TLE file to parse.
        mu (float): Gravitational parameter of the central body (m^3/s^2 or km^3/s^2).
        millenium (int): Millenium value to adjust the year from the TLE file.
    
    Returns:
        dict: A dictionary containing satellite information and computed orbital elements.
    """
    try:
        # Open and read TLE file
        with open(filename, 'r') as file:
            line1 = file.readline().split()
            line2 = file.readline().split()
    except IOError:
        raise IOError("Error opening the TLE file.")

    # Parse line 1
    sat = {}
    sat['satID'] = line1[1]
    sat['satDesignator'] = line1[2]
    sat['date'] = {}
    sat['date']['year'] = millenium + int(line1[3][:2])
    sat['date']['day'] = float(line1[3][2:])
    sat['dMeanMotion'] = float(line1[4])
    sat['ddMeanMotion'] = line1[5]  # Ignored here
    sat['Bstar'] = float('0.' + line1[6][:-2]) * 10 ** int(line1[6][-2:])
    sat['eph'] = line1[7]
    sat['elemNum'] = line1[8]

    # Parse line 2
    sat['Inclination'] = float(line2[2])
    sat['RAAN'] = float(line2[3])
    sat['Eccentricity'] = float('0.' + line2[4])
    sat['AOP'] = float(line2[5])
    sat['meanAnomaly'] = float(line2[6])
    sat['meanMotion'] = float(line2[7][:11])  # Mean motion in revs per day

    
    # Functions for Newton's method
    fun = lambda E: np.deg2rad(sat['meanAnomaly']) - E + sat['Eccentricity'] * np.sin(E)
    dfun = lambda E: -1 + sat['Eccentricity'] * np.cos(E)

    # Compute eccentric anomaly using Newton's method
    sat['EccenAnomaly'] = np.rad2deg(newton_root(fun, dfun, 0.0))

    # Compute true anomaly
    sat['TrueAnomaly'] = np.rad2deg(2. * np.arctan(np.sqrt((1. + sat['Eccentricity']) / (1. - sat['Eccentricity']))
                                                    * np.tan(np.deg2rad(sat['EccenAnomaly']) / 2.)))

    # Compute semimajor axis
    sat['SemimajorAxis'] = (mu * (86400 / sat['meanMotion'] / (2. * np.pi)) ** 2.) ** (1 / 3)

    return sat



def gravity_pointmass(r_vec, mu):
    """
    Computes the gravitational acceleration of a point mass orbiting another point mass or spherical body.
    
    Parameters:
        r_vec (np.array): Position vector of the orbiting point mass (3x1).
        mu (float): Gravitational constant for the central body.
    
    Returns:
        np.array: Acceleration vector (3x1) due to gravity.
    """
    r_norm = np.linalg.norm(r_vec)
    accel = -mu / r_norm**3 * r_vec
    
    return accel



def ode_pointmass(t, x, mu):
    """
    Computes the derivative of the state vector (position and velocity) of 
    a point mass orbiting a central body.
    
    Parameters:
        t (float): Time (not used in this calculation, included for ODE solver)
        x (np.array): State vector (6x1) where the first 3 elements are the 
                      position vector and the last 3 elements are the velocity vector.
        mu (float): Gravitational constant for the central body.
    
    Returns:
        np.array: Derivative of the state vector (6x1) where the first 3 elements 
                  are the velocity vector and the last 3 elements are the acceleration vector.
    """
    dxdt = np.zeros(6)
    
    # First 3 components are velocity (dx/dt)
    dxdt[0:3] = x[3:6]
    
    # Last 3 components are acceleration (dv/dt)
    dxdt[3:6] = gravity_pointmass(x[0:3], mu)
    
    return dxdt


def plotsphere(ax, R, N=100):
    """
    Plots a sphere on a given 3D axis using the given radius and number of points.
    
    Parameters:
        ax (Axes3D): Matplotlib 3D axis object where the sphere will be plotted.
        R (float): Radius of the sphere.
        N (int): Number of points for the sphere (default is 100).
    
    Returns:
        None
    """
    # Create the sphere's coordinates
    u, v = np.mgrid[0:2*np.pi:complex(N), 0:np.pi:complex(N)]
    sx = R * np.cos(u) * np.sin(v)
    sy = R * np.sin(u) * np.sin(v)
    sz = R * np.cos(v)
    
    # Set the color (as normalized RGB values)
    color = (0, 153/256, 153/256)  # Define a single RGB triplet

    # Plot the sphere on the provided axis
    ax.plot_surface(sx, sy, sz, color=color, edgecolor='none', zorder=0)
    
    return ax













