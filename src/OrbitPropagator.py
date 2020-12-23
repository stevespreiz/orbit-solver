# Orbit propagator class to encapsulate solve

import numpy as np
import matplotlib.pyplot as plt
## for newer scipy.integrate
# import scipy.integrate as ode
## for older scipy.integrate (in the videos)
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D

import planetary_data as pd

class OrbitPropagator:
    def __init__(self,r0,v0,tf,cb=pd.earth):
        self.r0 = r0
        self.v0 = v0
        self.tf = tf
        self.cb = cb
        self.ys = np.zeros((1,6))
        self.ts = np.zeros((1,1))


    def diff_eq(self,t,y):
        # unpack state
        rx, ry, rz, vx, vy, vz = y
        r = np.array([rx,ry,rz])

        # norm(r)
        norm_r = np.linalg.norm(r)

        # two-body acceleration
        ax,ay,az = -r*self.cb['mu']/norm_r**3

        return [vx,vy,vz,ax,ay,az]


    def propagate_orbit(self):
        ## Something wrong with this newer version of scipy.integrate so sticking
        ## with old one for now
        # # IC
        # self.y0 = self.r0+self.v0 # concatenate lists
        #
        # # set of t values at which output is desired
        #
        # t_output = np.linspace(0,self.tf,int(np.ceil(self.tf/100)))
        #
        # sol = ode.solve_ivp(self.diff_eq,[0,self.tf],self.y0,t_eval=t_output,method='LSODA')
        # self.ys = sol.y
        # self.ts = sol.t
        # self.ys = np.transpose(np.array(self.ys))

#########################################################################3
        # solver setup from the videos
        # time step
        dt = 100.0

        # total number of steps
        n_steps = int(np.ceil(self.tf/dt))

        # new dt
        dt = self.tf/n_steps

        # pre-allocate memory
        self.ys = np.zeros((n_steps,6))
        self.ts = np.zeros((n_steps,1))

        # IC
        y0 = self.r0+self.v0
        self.ys[0] = np.array(y0)
        n = 1 # time step

        # initialize solver
        solver = ode(self.diff_eq)
        solver.set_integrator('lsoda')
        solver.set_initial_value(y0,0)
        # solver.set_f_params(self.cb['mu'])

        # propogate orbit
        while solver.successful() and n < n_steps :
            solver.integrate(solver.t+dt)
            self.ts[n] = solver.t
            self.ys[n] = solver.y
            n += 1

        rs = self.ys[:,:3]


    # 3D plotting function
    def plot3D(self,show_plot=True,save_plot=False,plot_title='Default'):
        plt.style.use('dark_background')
        fig = plt.figure(figsize = (12,6))
        ax  = fig.add_subplot(111,projection='3d')
        rs = self.ys[:,:3]

        # plot trajectory
        ax.plot(rs[:,0],rs[:,1],rs[:,2],'w',label='trajectory')
        ax.plot([rs[0,0]],[rs[0,1]],[rs[0,2]],'wo',label='Initial position')

        # plot central body
        _u,_v = np.mgrid[0:2*np.pi:40j,0:np.pi:20j]
        _x  = self.cb['radius']*np.cos(_u)*np.sin(_v)
        _y = self.cb['radius']*np.sin(_u)*np.sin(_v)
        _z = self.cb['radius']*np.cos(_v)
        ax.plot_surface(_x,_y,_z,cmap='Blues')

        # plot x,y,z vectors
        l = self.cb['radius']*2
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

        ax.set_title(plot_title)
        plt.legend() # automatically fills earlier
        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(plot_title+'.png', dpi=300,edgecolor='none')
