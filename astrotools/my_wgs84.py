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
