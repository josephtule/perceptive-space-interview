import numpy as np
import matplotlib.pyplot as plt
import astrotools as at
import scipy as sc
import pandas as pd
import mplcursors

from mpl_toolkits.mplot3d import Axes3D

earth = at.MyWGS84("km")

# import data
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
                          df[df['ECEF'] == 'z']['velocity'].values]) / 1e4
clock_bias = np.array(df['clock'].values[0:-1:3])
date = df[df['ECEF'] == 'x']['time']

# Outlier clean up
min_leo = earth.SemimajorAxis + 150 # minimum LEO radius
max_geo = earth.SemimajorAxis + 35786 + 300 # maximum geo radius

sat['r_mag'] = np.linalg.norm(sat['r_itrf'], axis=0)
keep_ind = (sat['r_mag'] >= min_leo) & (sat['r_mag'] <= max_geo)
sat['r_itrf'] = sat['r_itrf'][:, keep_ind]
sat['v_itrf'] = sat['v_itrf'][:, keep_ind]
sat['r_mag'] = sat['r_mag'][keep_ind]
date = date[keep_ind]
clock_bias = clock_bias[keep_ind]

# Extract year, month, and day with fractional time
year = date.dt.year.to_numpy()
month = date.dt.month.to_numpy()
day = date.dt.day.to_numpy()
frac = ((date.dt.hour + date.dt.minute / 60 + date.dt.second / 3600) / 24).to_numpy()

CalUTtoJD_vec = np.vectorize(at.CalUTtoJD)
JD = CalUTtoJD_vec(year,month,day,frac*24)
GMST, _ = at.JDtoGMST(JD,0)
# print(GMST)


EOP2 = at.parseEOPFile("./astrotools/EOP2long.txt")
sat['r_j2000'] = np.zeros_like(sat['r_itrf'])
sat['v_j2000'] = np.zeros_like(sat['v_itrf'])
for i in range(len(sat['r_itrf'][0])):
    R_earthrot = at.rot(GMST[i], 3, "degrees").T
    R_nutation = at.RotNutation(JD[i], "UT1").T
    R_precession = at.RotPrecession(JD[i], "UT1").T
    R_polarmotion = at.RotPolarMotion(JD[i], EOP2).T
    full_rot = R_precession @ R_nutation @ R_earthrot @ R_polarmotion
    sat['r_j2000'][:,i] = full_rot @ sat['r_itrf'][:,i]
    sat['v_j2000'][:,i] = full_rot @ sat['v_itrf'][:,i]


# Enable hover functionality
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f"x: {sel.target[0]:.2f}, y: {sel.target[1]:.2f}"))


print("norms:")
print(sat['r_mag'][:6])




# Simulation and Filtering
dt = np.diff(JD) * 86400
day_skip = dt > dt[0] * 5
print(np.where(day_skip))


# plt.figure()
# plt.plot(dt)  
# plt.show()


## Extended Kalman filter parameters
# Initialize process noise covariance matrix (Q)
q1 = 1e-4
q2 = 1e-6
Q = np.diag([q1, q1, q1, q2, q2, q2])

# Initialize measurement noise covariance matrix (R)
r1 = 1e-4
r2 = 1e-8
R = np.diag([r1, r1, r1, r2, r2, r2])

# Initialize state covariance matrix (P)
# P_est = np.eye(6)
p_var = 10
v_var = .05
P_est = np.diag([p_var, p_var, p_var, v_var, v_var, v_var])


## Initialize estimation array
N = sat['r_mag'].shape[0]
x_est = np.zeros([6, N]);  
x_est[:, 0] = np.concatenate((sat['r_j2000'][:,0], sat['v_j2000'][:,0]))

print("cool")

dt_min = np.min(dt)


# gravity parameters
max_degree = 4
earth.read_egm('./astrotools/EGM2008_to2190_TideFree.txt', max_degree)


for k in range(N-1):
    # Prediction step
    r = np.linalg.norm(x_est[0:3, k])  # Compute the magnitude of the position vector

    # def f(t, x):
    #     r_vec = x[0:3]
    #     r = np.linalg.norm(r_vec)
        
    #     # Gravitational acceleration due to Earth's gravity (with J2 perturbation)
    #     J2 = 1.08263e-3  # J2 coefficient
    #     Re = earth.SemimajorAxis  # Earth's radius in km
    #     z2 = r_vec[2]**2
    #     r2 = r**2
        
    #     # Standard two-body gravitational force
    #     a_grav = -earth.mu / r**3 * r_vec
        
    #     # J2 perturbation acceleration
    #     a_J2 = (1.5 * J2 * earth.mu * Re**2 / r**5) * r_vec * (5 * z2 / r2 - 1)
    #     a_J2[2] = (1.5 * J2 * earth.mu * Re**2 / r**5) * r_vec[2] * (5 * z2 / r2 - 3)

    #     return np.concatenate((x[3:6], a_grav + a_J2))  # Return velocity and total acceleration

    f = lambda t,x: at.orbit_ode(t, x, 
                                 earth.mu, gravity="J", 
                                 max_degree=max_degree, Re=earth.SemimajorAxis, 
                                 C=earth.C, S=earth.S, GMST=GMST[k], J=earth.J)

    if dt[k] > dt_min * 1e1:
        ns = 100
    else:
        ns = 12
    _, x_pred = at.rk4_substeps(f, 0, x_est[:, k], dt[k], num_substeps=ns)

    # Jacobian (F_k) for state transition, initialization
    par2body = np.eye(3) * (-earth.mu / r**3)
    # Loop to modify the par2body matrix
    for i in range(3):
        for j in range(3):
            par2body[i, j] += 3 * earth.mu / r**5 * x_est[i, k] * x_est[j, k]

    # State transition Jacobian F
    F = dt[k] * np.block([[np.zeros((3, 3)), np.eye(3)], [par2body, np.zeros((3, 3))]])

    # Covariance prediction
    P_pred = F @ P_est @ F.T + Q

    # Kalman Gain
    H = np.eye(6)  # Measurement matrix
    S = H @ P_pred @ H.T + R  # Innovation covariance
    K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain

    # Measurement update
    z_k = np.concatenate((sat['r_j2000'][:,k+1], sat['v_j2000'][:,k+1]))

    innovation = z_k - H @ x_pred

    x_est[:,k+1] = x_pred + K @ innovation # Update state estimate
    P_est = (np.eye(6) - K @ H) @ P_pred # Covariance update


ax = plt.figure().add_subplot(projection="3d")
ax.plot(sat['r_j2000'][0],sat['r_j2000'][1],sat['r_j2000'][2],label="SV Trajectory",marker="*",linestyle='None')
# ax = at.plotsphere(ax,earth.SemimajorAxis)
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_zlabel("z [km]")
lim = 2*earth.SemimajorAxis
ax.set_xlim([-lim,lim])
ax.set_ylim([-lim,lim])
ax.set_zlim([-lim,lim])
ax.set_aspect('equal')
ax.plot(x_est[0,:],x_est[1,:],x_est[2,:],label="Estimated Trajectory",marker="o",linestyle="None")


fig, axes = plt.subplots(nrows=2, ncols=1)
axes[0].plot(JD, (x_est[0:3,:] - sat['r_j2000'][:,:]).T)
axes[1].plot(JD, (x_est[3:6,:] - sat['v_j2000'][:,:]).T)

plt.figure()
E = at.orbit_energy(sat['r_j2000'], sat['v_j2000'], earth.mu)
E_est = at.orbit_energy(x_est[0:3,:], x_est[4:6,:], earth.mu)
plt.plot(JD,E)
plt.plot(JD,E_est)

plt.show()
