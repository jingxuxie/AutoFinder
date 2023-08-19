#%%
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import scipy
from scipy.interpolate import interp1d

def predict_sio2(BGRcontr):
    contrast = np.load('E:/Desktop2023.1.17/AutoFinder/sio2_contrast.npy')
    z = np.linspace(80, 100, 201)
    err = np.sqrt((contrast[0] - BGRcontr[0])**2 + \
                  (contrast[1] - BGRcontr[1])**2 + \
                  (contrast[2] - BGRcontr[2])**2)

    predz = z[np.argmin(err)]
    
    try:
        lowerbound = min(z[err < min(err)*2])
        upperbound = max(z[err < min(err)*2])
    except:
        lowerbound, upperbound = 0, 0

    err = (upperbound - lowerbound) / 2

    return True, predz, err
# %%

if __name__ == '__main__':
    predict_sio2([0.15, 0.11, 0.1])
# %%
