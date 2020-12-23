# Driver for solver/plotting tests

import numpy as np
import planetary_data as pd
from OrbitPropagator import OrbitPropagator as OP

cb = pd.earth # central body

if __name__ == '__main__':
    # orbit parameters
    r_mag = cb['radius'] + 1500.0
    v_mag = np.sqrt(cb['mu']/r_mag)

    # initial state
    r0 = [r_mag,0,0]
    v0 = [0,v_mag,0]

    tf = 3600*60

    op = OP(r0,v0,tf)
    op.propagate_orbit()
    op.plot3D()
