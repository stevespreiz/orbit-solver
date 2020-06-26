# Solver for 2 body problem

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D


earth_radius = 6378.0 # km
earth_mu = 398600.0 # km^3/s^2

def diff_eq(t,y,mu):
    # unpack state
    rx, ry, rz, vx, vy, vz = y
    r = np.array([rx,ry,rz])

    # norm(r)
    norm_r = np.linalg.norm(r)

    # two-body acceleration
    ax,ay,az = -r*mu/norm_r**3

    return [vx,vy,vz,ax,ay,az]

#if __name__ == 'main':
# orbit parameters
r_mag = earth_radius + 500.0
v_mag = np.sqrt(earth_mu/r_mag)

# initial state
r0 = [r_mag,0,0]
v0 = [0,v_mag,0]

# time span
tf = 100*60.0

# time step
dt = 100.0

# total number of steps
n_steps = int(np.ceil(tf/dt))

# new dt
dt = tf/n_steps

# pre-allocate memory
ys = np.zeros((n_steps,6))
ts = np.zeros((n_steps,1))

# IC
y0 = r0+v0
ys[0] = np.array(y0)
n = 1 # time step

# initialize solver
solver = ode(diff_eq)
solver.set_integrator('lsoda')
solver.set_initial_value(y0,0)
solver.set_f_params(earth_mu)

# propogate orbit
while solver.successful() and n < n_steps :
    solver.integrate(solver.t+dt)
    ts[n] = solver.t
    ys[n] = solver.y
    n += 1

rs = ys[:,:3]

print(rs) # plotting function in second video
