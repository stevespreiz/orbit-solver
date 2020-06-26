# Solver for 2 body problem

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

# 3D plotting function
def plot(rs):
    fig = plt.figure(figsize = (12,6))
    ax  = fig.add_subplot(111,projection='3d')

    # plot trajectory
    ax.plot(rs[:,0],rs[:,1],rs[:,2],'w',label='trajectory')
    ax.plot([rs[0,0]],[rs[0,1]],[rs[0,2]],'wo',label='Initial position')

    # plot central body
    _u,_v = np.mgrid[0:2*np.pi:20j,0:np.pi:10j]
    _x  = earth_radius*np.cos(_u)*np.sin(_v)
    _y = earth_radius*np.sin(_u)*np.sin(_v)
    _z = earth_radius*np.cos(_v)
    ax.plot_surface(_x,_y,_z,cmap='Blues')

    # plot x,y,z vectors
    l = earth_radius*2
    x,y,z = [[0,0,0],[0,0,0],[0,0,0]]
    u,v,w = [[l,0,0],[0,l,0],[0,0,l]]

    ax.quiver(x,y,z,u,v,w,color='k')

    max_val = np.max(np.abs(rs))

    ax.set_xlim([-max_val,max_val])
    ax.set_ylim([-max_val,max_val])
    ax.set_zlim([-max_val,max_val])

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')

    ax.set_title('Orbit Around Earth')
    plt.legend() # automatically fills earlier
    plt.show()

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
r_mag = earth_radius
v_mag = np.sqrt(earth_mu/r_mag) + 1

# initial state
r0 = [r_mag,0,0]
v0 = [0,v_mag,0]

# time span
tf = 200*60.0

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

plot(rs) # plotting function in second video
