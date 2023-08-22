#%%
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import scipy
from scipy.interpolate import interp1d

def predict_sio2(bk_color):
    '''
    bk_color is taken at exposure 3000 and reference is at exposure 700
    with order of BGR
    '''
    
    BGRcontr = np.zeros(3)
    BGRcontr[0] = bk_color[0] / 150.365 / 0.95810 * 0.7 / 3
    BGRcontr[1] = bk_color[1] / 180.11 / 0.96294 * 0.7 / 3
    BGRcontr[2] = bk_color[2] / 196.45 / 0.95716 * 0.7 / 3

    # print(BGRcontr)
    absolute_path = os.path.dirname(__file__)
    relative_path = 'sio2_contrast.npy'
    file_path = os.path.join(absolute_path, relative_path)

    contrast = np.load(file_path)
    z = contrast[0]
    err = np.sqrt((contrast[1] - BGRcontr[0])**2 + \
                  (contrast[2] - BGRcontr[1])**2 + \
                  (contrast[3] - BGRcontr[2])**2)

    index = np.argmin(err)
    predz = z[index]
    
    try:
        lowerbound = min(z[err < min(err)*2])
        upperbound = max(z[err < min(err)*2])
    except:
        lowerbound, upperbound = 0, 0

    err = (upperbound - lowerbound) / 2

    return True, predz, index
# %%

if __name__ == '__main__':
    predict_sio2([0.15, 0.11, 0.1])
# %%
