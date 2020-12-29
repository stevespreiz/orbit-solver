# Driver for solver/plotting tests

import numpy as np
from math import sqrt #faster than np sqrt for scalars
import tools as t
import planetary_data as pd
from OrbitPropagator import OrbitPropagator as OP

cb = pd.earth # central body

if __name__ == '__main__':
    # orbit parameters
    r_mag = cb['radius'] + 1500.0
    v_mag = sqrt(cb['mu']/r_mag)

    # initial state
    r0 = [r_mag,0,0]
    v0 = [0,v_mag,0]

    # orbit parameters
    r_mag_1 = cb['radius'] + 400.0
    v_mag_1 = sqrt(cb['mu']/r_mag)

    # initial state
    r0_1 = [r_mag_1,0,0]
    v0_1 = [0,v_mag_1,5]

    tf = 3600*60

    op = OP(r0,v0,tf)
    op_1 = OP(r0_1,v0_1,tf)
    op.propagate_orbit()
    op_1.propagate_orbit()
    # op.plot3D()

    t.plot_n_orbits([op.rs,op_1.rs],labels=['0','1'])
