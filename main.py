import numpy as np
import matplotlib.pyplot as plt
import astrotools as at
import scipy as sc
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

plt.show()


