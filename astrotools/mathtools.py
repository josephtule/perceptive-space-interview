import math
import numpy as np

def newton_root(f, df, x0, tol=1e-7, max_iter=1000):
        x = x0
        iter = 0
        while iter < max_iter:
            x_next = x - f(x)/df(x)

            # tol check
            if abs(x_next - x) < tol:
                return x_next
            
            x = x_next
            iter += 1
        raise RuntimeError("Newton's method did not converge.")


def vectorangle(r1, r2, degorrad="degrees"):
    """
    Computes the angle between two vectors.

    Parameters:
        r1 (np.ndarray): First vector
        r2 (np.ndarray): Second vector
        degorrad (str): Flag indicating whether to return the angle in degrees or radians (default: "degrees")
    
    Returns:
        float: The angle between the two vectors (in degrees or radians)
    """
    # Compute magnitudes of the vectors
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    # Compute the angle in radians using the dot product formula
    angle = np.arccos(np.dot(r1, r2) / (r1_mag * r2_mag))

    # Convert the angle to degrees if necessary
    if degorrad.lower() in ["deg", "degrees", "degree", "degs"]:
        angle = np.rad2deg(angle)

    return angle



def rk4_step(f, t, x, dt):
    """
    Perform a single Runge-Kutta 4th order (RK4) step.
    
    Parameters:
    f  -- the function that defines the system's differential equations
    t  -- current time
    x  -- current state vector
    dt -- time step

    Returns:
    tnew -- updated time
    xnew -- updated state vector
    """
    k1 = f(t, x)
    k2 = f(t + dt / 2, x + dt * k1 / 2)
    k3 = f(t + dt / 2, x + dt * k2 / 2)
    k4 = f(t + dt, x + dt * k3)
    
    xnew = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    tnew = t + dt
    
    return tnew, xnew


def rk4_substeps(f, t, x, dt, num_substeps=10):
    sub_dt = dt / num_substeps  # Subdivide the time step
    for _ in range(num_substeps):
        t, x = rk4_step(f, t, x, sub_dt)
    return t, x


















