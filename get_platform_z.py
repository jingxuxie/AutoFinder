#%%
import numpy as np
from scipy.interpolate import RectBivariateSpline

#%%
x = np.linspace(0, 15000*5, 6)
y = np.linspace(0, 15000*5, 6)
Z = np.array([[-1600.,  6400.,  9600., 16000., 19200., 24000.],
                [-1600.,  4800.,  9600., 14400., 17600., 22400.],
                [    0.,  4800.,  6400., 11200., 12800., 17600.],
                [-3200.,  1600.,  4800.,  8000.,  9600., 12800.],
                [-1600.,     0.,  1600.,  1600.,  1600.,  3200.],
                [-4800., -3200., -1600., -1600., -1600., -1600.]])

f = RectBivariateSpline(x, y, Z, kx = 1, ky = 1)


def get_platform_z_diff(x1, y1, x2, y2):
    z1 = f.ev(x1, y1)
    z2 = f.ev(x2, y2)
    return z2 - z1
# %%
