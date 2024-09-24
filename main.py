import numpy as np
import matplotlib.pyplot as plt
import astrotools as at
import scipy as sc
import pandas as pd
import mplcursors

from mpl_toolkits.mplot3d import Axes3D


# x = np.array([1,2,3])
# y = np.array([5,6,7])

# plt.plot(x,y)
# plt.show()
# print(at.CalUTtoJD(1999,9,10,.5))
# print(at.dayOfYearToMonthDay(2024, 205.5))
# print(at.fracDaytoHMS(0.256))
R = at.rot(45,3,"deg")
# print(R)

# print(np.dot(R,x))
# R2, _ = at.EAtoDCM(np.array([1,2,3]),np.array([3,2,1]),"degrees")
# print(R2)
# obj = {
#     "SemimajorAxis": 6.796593878694531e+03,
#     "Eccentricity": 7.373000000000000e-04,
#     "Inclination": 51.638800000000003,
#     "RAAN": 2.005949000000000e+02,
#     "AOP": 17.026399999999999,
#     "TrueAnomaly": 83.057698617457802}
earth = at.MyWGS84("km")

arms = {'geod': np.array([40.43113274911836,
                           -86.91497485764418,
                             0])}
arms['r_itrf'] = at.GEODtoECEF(arms['geod'][0],
                               arms['geod'][1],
                               arms['geod'][2],
                               earth)

print("arms_ecef:",arms['r_itrf'])
obj = at.parseTLE("ISS_TLE.txt",earth.mu,2000)

obj["date"]["month"], obj["date"]["day"], obj["date"]["frac"] = at.dayOfYearToMonthDay(obj["date"]["year"],obj["date"]["day"])

print(obj['date'])

JD_start = at.CalUTtoJD(obj['date']['year'],
                  obj['date']['month'],
                  obj['date']['day'],
                  at.timeconverter(obj['date']['frac'], "UTC", "UT", "frac")*24)

N_time = 360
orbit_period = 2*np.pi*np.sqrt(obj['SemimajorAxis']**3/earth.mu)
JD = np.linspace(JD_start, JD_start + orbit_period/24/60/60, N_time)
np.set_printoptions(precision=15)

GMST, arms['LMST'] = at.JDtoGMST(JD, arms["geod"][1])

obj["rv_j2000"] = at.COEtoRV(obj["SemimajorAxis"],
                 obj["Eccentricity"],
                 obj["Inclination"],
                 obj["RAAN"],
                 obj["AOP"],
                 obj["TrueAnomaly"],
                 earth.mu,
                 "deg")

dt = orbit_period/N_time
t0 = 0
tf = orbit_period
t_eval = np.linspace(t0, tf, N_time)
abs_tol = 1e-12
rel_tol = 1e-12

odesol = sc.integrate.solve_ivp(at.ode_pointmass, (t0, tf), obj['rv_j2000'], args=(earth.mu,), 
                     t_eval=t_eval, rtol=rel_tol, atol=abs_tol, max_step=dt, method="RK45")
obj['r_j2000'] = np.array(odesol.y[:3])
obj['v_j2000'] = np.array(odesol.y[3:])
ax = plt.figure().add_subplot(projection="3d")


ax.plot(obj['r_j2000'][0],obj['r_j2000'][1],obj['r_j2000'][2],zorder=1)
ax = at.plotsphere(ax,earth.SemimajorAxis)

ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_zlabel("z [km]")
ax.set_aspect('equal')
ax.set_title("ISS Orbit")


df = pd.read_csv('GPS meas.csv')
df['time'] = pd.to_datetime(df['time'])

# sort by x, y, z
xyz_order = {'x': 1, 'y': 2, 'z': 3}
df['xyz_order'] = df['ECEF'].map(xyz_order)
df = df.sort_values(by=['time', 'xyz_order'])
df = df.drop(columns=['xyz_order'])

# extract x, y, z positions and velocities
sat = {}
sat['r_itrf'] = np.array([df[df['ECEF'] == 'x']['position'].values,
                          df[df['ECEF'] == 'y']['position'].values,
                          df[df['ECEF'] == 'z']['position'].values])
sat['v_itrf'] = np.array([df[df['ECEF'] == 'x']['velocity'].values,
                          df[df['ECEF'] == 'y']['velocity'].values,
                          df[df['ECEF'] == 'z']['velocity'].values])
clock_bias = np.array(df['clock'].values[0:-1:3])

date = df[df['ECEF'] == 'x']['time']

# Extract year, month, and day with fractional time
year = date.dt.year.to_numpy()
month = date.dt.month.to_numpy()
day = date.dt.day.to_numpy()
frac = ((date.dt.hour + date.dt.minute / 60 + date.dt.second / 3600) / 24).to_numpy()

CalUTtoJD_vec = np.vectorize(at.CalUTtoJD)
JD = CalUTtoJD_vec(year,month,day,at.timeconverter(frac*24,"UTC","UT","hr"))
GMST, _ = at.JDtoGMST(JD,0)
print(GMST)


EOP2 = at.parseEOPFile("./astrotools/EOP2long.txt")
sat['r_j2000'] = np.zeros_like(sat['r_itrf'])
sat['v_j2000'] = np.zeros_like(sat['v_itrf'])
for i in range(len(sat['r_itrf'][0])):
    R_earthrot = at.rot(GMST[i], 3, "degrees").T
    R_nutation = at.RotNutation(JD[i], "UT1").T
    R_precession = at.RotPrecession(JD[i], "UT1").T
    R_polarmotion = at.RotPolarMotion(JD[i], EOP2).T
    full_rot = R_precession * R_nutation * R_earthrot * R_polarmotion
    sat['r_j2000'][:,i] = full_rot @ sat['r_itrf'][:,i]
    sat['v_j2000'][:,i] = full_rot @ sat['v_itrf'][:,i]

# temp remove outliers
sat['r_j2000'][np.abs(sat['r_j2000']) > 1e8] = 0

# plt.plot(JD,sat['r_j2000'][0,:],label="x")
# plt.plot(JD,sat['r_j2000'][1,:],label="y")
# plt.plot(JD,sat['r_j2000'][2,:],label="z")
ax = plt.figure().add_subplot(projection="3d")
ax.plot(sat['r_j2000'][0],sat['r_j2000'][1],sat['r_j2000'][2])
ax = at.plotsphere(ax,earth.SemimajorAxis)
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_zlabel("z [km]")
ax.set_aspect('equal')


# Enable hover functionality
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f"x: {sel.target[0]:.2f}, y: {sel.target[1]:.2f}"))


plt.show()

