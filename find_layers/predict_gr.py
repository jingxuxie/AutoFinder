#%%
import numpy as np
import matplotlib.pyplot as plt
import re
import os
#%%
def predict_gr(BGRcontr):
    folder = os.path.dirname(__file__.replace('\\', '/'))
    data_points = np.load(folder + '/contrast_gr.npy')
    distance = np.sqrt(np.sum((data_points - BGRcontr)**2, axis = 1))
    
    BGRcontr = np.array(BGRcontr)

    b = -np.array([0.000564003245235811, -0.00337583575774649, 0.0110656614062229])
    g = -np.array([0.00160727422793472, 0.0124299602515373, -0.224158130974298])
    r = -np.array([0.00192042964436906, 0.000962550252638925, -0.195638618531615])

    z = np.linspace(0, 10, 1001)

    B, G, R = 0, 0, 0
    for i in range(3):
        B += b[i] * z**(3 - i)
        G += g[i] * z**(3 - i)
        R += r[i] * z**(3 - i)
    
    err = np.sqrt((B - BGRcontr[0])**2 + (G - BGRcontr[1])**2 + (R - BGRcontr[2])**2)
    
    # print(np.min(np.min(err)))
    if np.min([np.min(distance), np.min(err)]) > 0.15:
        return False, 0, 0

    predz = z[np.argmin(err)]
    # err /= 0.05
    try:
        lowerbound = min(z[err < min(err)*2])
        upperbound = max(z[err < min(err)*2])
    except:
        lowerbound, upperbound = 0, 0

    err = (upperbound - lowerbound) / 2

    return True, predz, err


def predict_gr(BGRcontr, SiO2_thickness = 95):
    contrast = - np.load('E:/Desktop2023.1.17/AutoFinder/autofinder/find_layers/gr_contrast.npy')
    z = contrast[0]
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


#%%
if __name__ == '__main__':
    
    r = np.array([0.00192042964436906, 0.000962550252638925, -0.195638618531615])
    g = np.array([0.00160727422793472, 0.0124299602515373, -0.224158130974298])
    b = np.array([0.000564003245235811, -0.00337583575774649, 0.0110656614062229])

    z = np.linspace(0, 10, 1001)

    R, G, B = 0, 0, 0
    for i in range(3):
        R += r[i] * z**(3 - i)
        G += g[i] * z**(3 - i)
        B += b[i] * z**(3 - i)

    plt.figure()
    plt.plot(z, -R, color = 'r')
    plt.plot(z, -G, color = 'g')
    plt.plot(z, -B, color = 'b')


    with open('E:\Desktop2023.1.17\AutoFinder\matlab contast code\old contrast_gr.txt') as f:
        lines = f.readlines()

    old_contrast = []
    new_contrast = []
    for line in lines:
        char = re.split(r'\]|\[| ', line)
        old_contrast.append(np.array([float(char[i + 1]) for i in range(4)]))
        new_contrast.append(np.array([float(char[i + 7]) for i in range(4)]))

    old_contrast = np.array(old_contrast)
    new_contrast = np.array(new_contrast)
    new_contrast = np.flip(new_contrast[:, :3], axis = 1)
    # plt.figure()
    # plt.scatter(new_contrast[:, 1], new_contrast[:, 3])

    # plt.figure()
    for contr in new_contrast:
        ret, z, err = predict_gr(-contr)
        if not ret:
            print('not in range')
        plt.errorbar(z, -contr[0], xerr = err, color = 'b')
        plt.errorbar(z, -contr[1], xerr = err, color = 'g')
        plt.errorbar(z, -contr[2], xerr = err, color = 'r')
# %%
