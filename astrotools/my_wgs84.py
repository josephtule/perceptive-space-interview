import numpy as np
from astrotools import norm_factor

class MyWGS84:
    """
    Generates default values for Earth using the WGS84 ellipsoid model.
    
    Attributes:
        mu (float): Earth's gravitational parameter (m^3/s^2 or km^3/s^2 depending on units)
        omega (float): Earth's angular velocity magnitude (rad/s)
        semi_major_axis (float): Semi-major axis of the Earth (meters or kilometers)
        flattening (float): Flattening of the Earth
    """
    
    def __init__(self, units="meters"):
        """
        Initializes the MyWGS84 object with the desired units (meters or kilometers).

        Args:
            units (str): The units to use for the Earth's properties ("meters" or "kilometers")
        """
        self.units = units.lower()  # Handle case insensitivity

        # Use WGS84 model (default properties for Earth in meters)
        self.mu = 3.986004418e14  # Gravitational parameter [m^3/s^2]
        self.omega = 7.292115e-5  # Angular velocity [rad/s]
        self.SemimajorAxis = 6378137.0 # Earth Semimajor axis [m]
        self.SemiminorAxis = 6356752.31424518 # Earth Semiminor axis [m]
        self.InverseFlattening = 298.257223563
        self.Eccentricity = 0.0818191908426215
        self.MeanRadius = 6.371008771415059e+06
        self.Flattening = 1. / self.InverseFlattening
        self.ThirdFlattening = 0.001679220386384
        self.SurfaceArea = 5.100656217240886e+14 # surface area of the Earth [m^2]
        self.Volume = 1.083207319801408e+21 # volume of the Earth [m^3]

        
        # Convert to kilometers if required
        if self.units in ["km", "kilometers", "kilometer"]:
            self.mu = self.mu / 1e3**3  # Convert mu to km^3/s^2
            self.SemimajorAxis = self.SemimajorAxis / 1e3
            self.SemiminorAxis = self.SemiminorAxis / 1e3
            self.MeanRadius = self.MeanRadius / 1e3
            self.SurfaceArea = self.SurfaceArea / 1e3**2
            self.Volume = self.Volume / 1e3**3
            
    def read_egm(self, filepath, max_deg):
        """
        Reads the EGM2008 gravity model file and assigns the C and S coefficients 
        along with their uncertainties up to max_deg as attributes of the class.

        Parameters:
        filepath (str)   -- Path to the EGM2008 file
        max_deg (int)    -- Maximum degree for the spherical harmonics

        Attributes:
        C (np.array)     -- C coefficients array (max_deg+1, max_deg+1)
        S (np.array)     -- S coefficients array (max_deg+1, max_deg+1)
        C_uncert (np.array) -- Uncertainty in C coefficients (max_deg+1, max_deg+1)
        S_uncert (np.array) -- Uncertainty in S coefficients (max_deg+1, max_deg+1)
        """
        k = 1
        self.C = np.zeros((max_deg + k, max_deg + k))       # Initialize C array
        self.S = np.zeros((max_deg + k, max_deg + k))       # Initialize S array
        self.C_uncert = np.zeros((max_deg + k, max_deg + k))  # Initialize uncertainty for C
        self.S_uncert = np.zeros((max_deg + k, max_deg + k))  # Initialize uncertainty for S
        self.J = np.zeros((max_deg+k,1))
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
                self.C[n, m] = C_value
                self.S[n, m] = S_value
                self.C_uncert[n, m] = C_uncert_value
                self.S_uncert[n, m] = S_uncert_value
                
        
            for i in range(max_deg+k):
                self.J[i] = self.C[i,0] * norm_factor(i,0)