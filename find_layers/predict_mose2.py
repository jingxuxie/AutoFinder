#%%
import numpy as np


#%%
def predict_mose2(BGRcontr, SiO2_thickness = 90):
    contrast = - np.load('E:/Desktop2023.1.17/AutoFinder/autofinder/find_layers/mose2_contrast.npy')
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
