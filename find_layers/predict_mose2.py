#%%
import numpy as np
import os


#%%
def predict_mose2(BGRcontr, bk_color = None):

    absolute_path = os.path.dirname(__file__)
    relative_path = 'mose2_contrast.npy'
    file_path = os.path.join(absolute_path, relative_path)
    contrast = - np.load(file_path)
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
