import numpy as np
import matplotlib.pyplot as plt
import planetary_data as pd
from mpl_toolkits.mplot3d import Axes3D

# 3D plotting function
def plot_n_orbits(rs,labels,cb=pd.earth,show_plot=True,save_plot=False,plot_title='N Orbits'):
    plt.style.use('dark_background')
    fig = plt.figure(figsize = (12,6))
    ax  = fig.add_subplot(111,projection='3d')


    # plot trajectory
    n = 0
    for r in rs:
        ax.plot(r[:,0],r[:,1],r[:,2],label=labels[n])
        ax.plot([r[0,0]],[r[0,1]],[r[0,2]])
        n = n+1

    # plot central body
    _u,_v = np.mgrid[0:2*np.pi:40j,0:np.pi:20j]
    _x  = cb['radius']*np.cos(_u)*np.sin(_v)
    _y = cb['radius']*np.sin(_u)*np.sin(_v)
    _z = cb['radius']*np.cos(_v)
    ax.plot_surface(_x,_y,_z,cmap='Blues')

    # plot x,y,z vectors
    l = cb['radius']*2
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
