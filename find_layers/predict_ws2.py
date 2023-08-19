import numpy as np
import matplotlib.pyplot as plt
import re
import os
#%%
def predict_ws2(BGRcontr):
    contrast_range = np.array([[-0.05, 0.05], [-0.01, 0.13], [0.09, 0.21]])
    for i in range(3):
        if not contrast_range[i, 0] <= BGRcontr[i] <= contrast_range[i, 1]:
            return False, 0, 0
    return True, 0.7, 0

def predict_ws2(BGRcontr, SiO2_thickness = 90):
    contrast = - np.load('E:/Desktop2023.1.17/AutoFinder/autofinder/find_layers/ws2_contrast.npy')
    z = -contrast[0]
    err = np.sqrt((contrast[1] - BGRcontr[0])**2 + \
                  (contrast[2] - BGRcontr[1])**2 + \
                  (contrast[3] - BGRcontr[2])**2)

    predz = z[np.argmin(err)]
    
    try:
        lowerbound = min(z[err < min(err)*2])
        upperbound = max(z[err < min(err)*2])
    except:
        lowerbound, upperbound = 0, 0

    err = (upperbound - lowerbound) / 2

    return True, predz, err
# %%
