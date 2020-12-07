# Orbit propagator class to encapsulate solve

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

import planetary_data as pd

class OrbitPropagator:
    def __init__(self,r0,v0,tf,dt,cb=pd.earth):
        self.r0 = r0
        self.v0 = v0
        self.tf = tf
        self.dt = dt
        self.cb = cb

    def propagate_orbit(self):
        # total number of steps
        self.n_steps = int(np.ceil(self.tf/self.dt))

        # new dt
        self.dt = self.tf/self.n_steps

        # pre-allocate memory
        self.ys = np.zeros((self.n_steps,6))
        self.ts = np.zeros((self.n_steps,1))

        # IC
        self.y0 = self.r0+self.v0 # concatenate lists
        self.ys[0] = np.array(self.y0)
        self.n = 1 # time step

        # initialize solver
        solver = integrate.RK45(diff_eq,0,y0,tf)

        # propogate orbit
        while solver.successful() and n < n_steps :
            solver.integrate(solver.t+dt)
            ts[n] = solver.t
            ys[n] = solver.y
            n += 1

        rs = ys[:,:3]

    def diff_eq(t,y):
        # unpack state
        rx, ry, rz, vx, vy, vz = y
        r = np.array([rx,ry,rz])

        # norm(r)
        norm_r = np.linalg.norm(r)

        # two-body acceleration
        ax,ay,az = -r*self.cb['mu']/norm_r**3

        return [vx,vy,vz,ax,ay,az]
