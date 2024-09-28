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


def orbit_ode(t, x, mu, gravity="newton", perturbations=None, Re=None, J=None, C=None, S=None, GMST=None, max_degree=None):
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
    if gravity.lower() in ['newton','newtonian','pointmass','spherical','point','central']:
        dxdt[3:6] = gravity_pointmass(x[0:3], mu)
    elif gravity.lower() in ['j','j2','j3','j4','j5','j6']:
        dxdt[3:6] = gravity_pointmass(x[0:3], mu)
        dxdt[3:6] += gravity_zonal(x[0:3], Re, mu, J, max_degree)
    elif gravity.lower() in ['sphericalharmonic','sphericalharmonic','sphharmonic','sphharmonics','sphharm','sphharmon']:
        dxdt[3:6] += rot(GMST, 3, "degrees") @ gravity_sphharm(x[0:3], Re, mu, max_degree, C, S)


    return dxdt





def orbit_energy(r, v, mu):
    r = np.linalg.norm(r, axis=0)
    v = np.linalg.norm(v, axis=0)
    
    return (v**2)/2 - mu / r


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


def gravity_zonal(r, Re, mu, J, max_degree):
    """Computes acceleration due to gravity due to the oblateness of the earth

    Args:
        r (np.array): position vector
        mu (float): central body gravitational parameter
        J (np.array): array of Ji coefficients
        
    Returns:
        np.array: acceleration vector
    """
    
    a = np.zeros(3)
    
    if max_degree == None:
        max_degree = len(J) + 1
    
    r_mag = np.linalg.norm(r)
    mur2 = mu / r_mag**2
    Rer = Re / r_mag
    zr = r[2] / r_mag
    
    if max_degree == 2:
        J2 = J[2]
        a += -3./2. * J2 * mur2 * Rer**2 * np.array([1 - 5 * zr**2,
                                                     1 - 5 * zr**2,
                                                     3 - 5 * zr**2]) * r/r_mag
    if max_degree == 3:
        J3 = J[3]
        a += -1./2. * J3 * mur2 * Rer**3 * np.array([5 * (3 * zr - 7 * zr**3) * r[0]/r_mag,
                                                     5 * (3 * zr - 7 * zr**3) * r[1]/r_mag,
                                                     -(3 - 30 * zr**2 + 35 * zr**4)])
    if max_degree == 4:
        J4 = J[4]
        a +=  5./8. * J4 * mur2 * Rer**4 * np.array([3 * (1 - 14 * zr**2 + 21 * zr**4),
                                                     3 * (1 - 14 * zr**2 + 21 * zr**4),
                                                     (15 - 70 * zr**2 + 63 * zr**4)]) * r/r_mag
    if max_degree == 5:
        J5 = J[5]
        a +=  3./8. * J5 * mur2 * Rer**5 * np.array([7 * (5 * zr - 30 * zr**3 + 33 * zr**5) * r[0]/r_mag,
                                                     7 * (5 * zr - 30 * zr**3 + 33 * zr**5) * r[1]/r_mag,
                                                     -(5 - 105 * zr**2 + 315 * zr**4 - 231 * zr**6)])
    if max_degree == 6:
        J6 = J[6]
        a += -7./16 * J6 * mur2 * Rer**6 * np.array([(5 - 135 * zr**2 + 495 ** zr**4 - 429 * zr**6),
                                                     (5 - 135 * zr**2 + 495 ** zr**4 - 429 * zr**6),
                                                     (35 - 315 * zr**2 + 693 * zr**4 - 429 * zr**6)]) * r/r_mag
    
    return a


def gravity_sphharm(x, Re, mu, max_degree, C, S):
    """
    Calculates gravity with spherical harmonics, outputs in ECEF.
    
    Parameters:
    x -- Position vector in ECEF coordinates [x, y, z]
    earth -- Object with Earth's constants, such as Re (Earth's radius), mu (gravitational constant), maxdeg, C, S
    
    Returns:
    g -- Gravitational acceleration vector in ECEF coordinates
    """
    r = np.linalg.norm(x)
    r2 = r**2
    sqrtx1x2 = np.sqrt(x[0]**2 + x[1]**2)
    
    # Compute geocentric latitude
    phi = np.arcsin(x[2] / r)
    
    # Calculate normalized Legendre polynomials
    P, scaleFactor = normlegendre(phi, max_degree)
    
    # Compute longitude (lambda)
    lambda_ = np.arctan2(x[1], x[0])
    slambda = np.sin(lambda_)
    clambda = np.cos(lambda_)
    
    smlambda = np.zeros(max_degree + 1)
    cmlambda = np.zeros(max_degree + 1)
    
    smlambda[0] = 0
    cmlambda[0] = 1
    smlambda[1] = slambda
    cmlambda[1] = clambda
    
    for m in range(2, max_degree + 1):
        smlambda[m] = 2.0 * clambda * smlambda[m - 1] - smlambda[m - 2]
        cmlambda[m] = 2.0 * clambda * cmlambda[m - 1] - cmlambda[m - 2]
    
    rRatio = Re / r
    rRatio_n = rRatio
    
    # Initialize summation of gravity in radial coordinates
    dUdrSumN = 1
    dUdphiSumN = 0
    dUdlambdaSumN = 0
    
    # Summation of gravity in radial coordinates
    for n in range(2, max_degree + 1):
        rRatio_n *= rRatio
        
        dUdrSumM = np.zeros(max_degree + 1)
        dUdphiSumM = np.zeros(max_degree + 1)
        dUdlambdaSumM = np.zeros(max_degree + 1)
        
        for m in range(n + 1):
            dUdrSumM[m] = P[n, m] * (C[n, m] * cmlambda[m] + S[n, m] * smlambda[m])
            dUdphiSumM[m] = (P[n, m + 1] * scaleFactor[n, m] - x[2] / sqrtx1x2 * m * P[n, m]) * (C[n, m] * cmlambda[m] + S[n, m] * smlambda[m])
            dUdlambdaSumM[m] = m * P[n, m] * (S[n, m] * cmlambda[m] - C[n, m] * smlambda[m])
        
        dUdrSumN += np.sum(dUdrSumM) * rRatio_n * (n+1)
        dUdphiSumN += np.sum(dUdphiSumM) * rRatio_n
        dUdlambdaSumN += np.sum(dUdlambdaSumM) * rRatio_n
    
    # Gravity in spherical coordinates
    dUdr = -mu / r2 * dUdrSumN
    dUdphi = mu / r * dUdphiSumN
    dUdlambda = mu / r * dUdlambdaSumN
    
    # Gravity in ECEF coordinates
    g = np.zeros(3) 
    g[0] = (1 / r * dUdr - x[2] / (r2 * sqrtx1x2) * dUdphi) * x[0] - (dUdlambda / (x[0]**2 + x[1]**2)) * x[1]
    g[1] = (1 / r * dUdr - x[2] / (r2 * sqrtx1x2) * dUdphi) * x[1] + (dUdlambda / (x[0]**2 + x[1]**2)) * x[0]
    g[2] = 1 / r * dUdr * x[2] + (sqrtx1x2) / (r2) * dUdphi
    
    # Special case for poles
    if abs(np.arctan2(x[2], sqrtx1x2)) == np.pi / 2:
        g = np.array([0, 0, (1 / r) * dUdr * x[2]])
    
    return g


def normlegendre(phi, max_degree):
    """
    Computes the normalized Legendre polynomials.
    
    Parameters:
    phi -- Geocentric latitude
    maxdeg -- Maximum degree of spherical harmonics
    
    Returns:
    P -- Normalized associated Legendre polynomials
    scaleFactor -- Scaling factors for the polynomials
    """
    P = np.zeros((max_degree + 3, max_degree + 3))
    scaleFactor = np.zeros((max_degree + 3, max_degree + 3))
    
    cphi = np.array(np.cos(np.pi / 2 - phi))
    sphi = np.array(np.sin(np.pi / 2 - phi))
    
    cphi[np.abs(cphi) <= np.finfo(float).eps] = 0
    sphi[np.abs(sphi) <= np.finfo(float).eps] = 0
    
    P[0, 0] = 1
    sqrt3 = np.sqrt(3)
    P[1, 0] = sqrt3 * cphi
    scaleFactor[0, 0] = 0
    scaleFactor[1, 0] = 1
    P[1, 1] = sqrt3 * sphi
    scaleFactor[1, 1] = 0
    
    for n in range(2, max_degree + 3):
        k = n
        sqrt2n1 = np.sqrt(2 * n + 1)
        
        for m in range(n + 1):
            p = m
            if n == m:
                P[k, k] = sqrt2n1 / np.sqrt(2 * n) * sphi * P[k - 1, k - 1]
                scaleFactor[k, k] = 0
            elif m == 0:
                P[k, p] = sqrt2n1 / n * (np.sqrt(2 * n - 1) * cphi * P[k - 1, p] - (n - 1) / np.sqrt(2 * n - 3) * P[k - 2, p])
                scaleFactor[k, p] = np.sqrt((n + 1) * n / 2)
            else:
                P[k, p] = sqrt2n1 / (np.sqrt(n + m) * np.sqrt(n - m)) * (
                    np.sqrt(2 * n - 1) * cphi * P[k - 1, p] -
                    np.sqrt(n + m - 1) * np.sqrt(n - m - 1) / np.sqrt(2 * n - 3) * P[k - 2, p])
                scaleFactor[k, p] = np.sqrt((n + m + 1) * (n - m))
    
    return P, scaleFactor


def norm_factor(l, m):
    
    def delta(m):
        if m == 0:
            return 1
        else:
            return 2

    return math.sqrt(math.factorial(l - m) * delta(m) * (2 * l + 1) / math.factorial(m + l))


def read_egm(filepath, max_deg):
    """
    Reads the EGM2008 gravity model file and returns the C and S coefficients 
    along with their uncertainties up to max_deg.

    Parameters:
    filepath (str)   -- Path to the EGM2008 file
    max_deg (int)    -- Maximum degree for the spherical harmonics

    Returns:
    C (np.array)     -- C coefficients array (max_deg+1, max_deg+1)
    S (np.array)     -- S coefficients array (max_deg+1, max_deg+1)
    C_uncert (np.array) -- Uncertainty in C coefficients (max_deg+1, max_deg+1)
    S_uncert (np.array) -- Uncertainty in S coefficients (max_deg+1, max_deg+1)
    """
    n = 1
    C = np.zeros((max_deg + n, max_deg + n))       # Initialize C array
    S = np.zeros((max_deg + n, max_deg + n))       # Initialize S array
    C_uncert = np.zeros((max_deg + n, max_deg + n))  # Initialize uncertainty for C
    S_uncert = np.zeros((max_deg + n, max_deg + n))  # Initialize uncertainty for S

    with open(filepath, 'r') as f:
        for line in f:
            # Split the line into fields based on spaces
            data = line.split()
            n = int(data[0])  # Degree
            m = int(data[1])  # Order
            
            if n > max_deg:  # Stop if the degree exceeds the max degree
                break

            # Extract values, replace Fortran 'D' with Python 'E' for exponents
            C_value = float(data[2].replace('D', 'E'))
            S_value = float(data[3].replace('D', 'E'))
            C_uncert_value = float(data[4].replace('D', 'E'))
            S_uncert_value = float(data[5].replace('D', 'E'))

            # Assign values to the arrays
            C[n, m] = C_value
            S[n, m] = S_value
            C_uncert[n, m] = C_uncert_value
            S_uncert[n, m] = S_uncert_value

    return C, S, C_uncert, S_uncert


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