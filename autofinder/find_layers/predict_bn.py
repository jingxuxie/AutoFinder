#%%
import numpy as np
import matplotlib.pyplot as plt
import re
import os
#%%
def predict_bn(BGRcontr):
    folder = os.path.dirname(__file__.replace('\\', '/'))
    data_points = np.load(folder + '/contrast_bn.npy')
    distance = np.sqrt(np.sum((data_points - BGRcontr)**2, axis = 1))
    
    BGRcontr = np.array(BGRcontr)

    b = np.array([-2.04504163e-07, -2.04804419e-05,  3.09080901e-03, -1.06792139e-01])
    g = np.array([-3.06745575e-06,  2.83331597e-04, -8.03670007e-03,  2.34471889e-02])
    r = np.array([-4.09133508e-07,  8.36883163e-05, -4.90582533e-03,  5.81517490e-02])

    z = np.linspace(0, 50, 501)

    B, G, R = 0, 0, 0
    degree = 4
    for i in range(degree):
        B += b[i] * z**(degree - i)
        G += g[i] * z**(degree - i)
        R += r[i] * z**(degree - i)
    
    err = np.sqrt((B - BGRcontr[0])**2 + (G - BGRcontr[1])**2 + (R - BGRcontr[2])**2)
    # print(np.min([np.min(distance), np.min(err)]))
    if np.min([np.min(distance), np.min(err)]) > 0.4:
        return False, 0, 0
    
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
    with open('E:/Desktop2023.1.17/AutoFinder/new dark.txt') as f:
        lines = f.readlines()
    new_contrast_dark = []
    for line in lines:
        char = re.split(r'\]|\[| ', line)
        new_contrast_dark.append(np.array([float(char[i + 1]) for i in range(4)]))

    new_contrast_dark = np.array(new_contrast_dark)[:, :3]
    new_contrast_dark = np.flip(new_contrast_dark, axis = 1)
    plt.figure()
    for contr in new_contrast_dark:
        ret, z, err = predict_bn(-contr)
        if not ret:
            print(contr)
            print('not in range')
            continue
        plt.scatter(z, -contr[0], color = 'b')
        plt.scatter(z, -contr[1], color = 'g')
        plt.scatter(z, -contr[2], color = 'r')
# %%
