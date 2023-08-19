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