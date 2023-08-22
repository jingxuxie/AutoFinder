#%%
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import scipy
from scipy.interpolate import interp1d
from autofinder.find_layers.predict_sio2 import predict_sio2
from tmm.tmm_core import coh_tmm
#%%
'''
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
'''

def predict_bn(BGRcontr, bk_color):

    ret, _, index = predict_sio2(np.array(bk_color) * 3)
    
    absolute_path = os.path.dirname(__file__)
    relative_path = 'bn_contrast.npy'
    file_path = os.path.join(absolute_path, relative_path)
    contrast = - np.load(file_path)
    contrast = contrast[:, :, index]

    z = np.linspace(0, 500, 5001)
    err = np.sqrt((contrast[:, 0] - BGRcontr[0])**2 + \
                  (contrast[:, 1] - BGRcontr[1])**2 + \
                  (contrast[:, 2] - BGRcontr[2])**2)

    predz = z[np.argmin(err)]
    
    try:
        lowerbound = min(z[err < min(err)*2])
        upperbound = max(z[err < min(err)*2])
    except:
        lowerbound, upperbound = 0, 0

    err = (upperbound - lowerbound) / 2

    return True, predz, err


def predict_thin_bn(BGRcontr, bk_color):
    return predict_bn(BGRcontr, np.array(bk_color)/3)


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
'''
data = scipy.io.loadmat('E:/Desktop2023.1.17/AutoFinder/matlab bn/refractiveindex.mat')
wavelength = data['lambda'].squeeze()
nk_si = data['n_Si']
nk_sio2 = data['n_SiO2']
nk_bn = data['n_BN']

nk_fn_si = interp1d(wavelength, nk_si, )
nk_fn_sio2 = interp1d(wavelength, nk_sio2, kind='linear')
nk_fn_bn = interp1d(wavelength, nk_bn, )

z = np.linspace(0, 500, 5001)
lambda_list = [452, 537, 633]
color_list = ['b', 'g', 'r']
contrast = np.zeros((4, len(z)))
contrast[0] = z
for i in range(len(lambda_list[:3])):
    lambda_vac = lambda_list[i]
    n_list = [1,  nk_fn_bn(lambda_vac), nk_fn_sio2(lambda_vac), nk_fn_si(lambda_vac)]
    R_list = [[] for i in range(3)]

    for thickness in z:
        d_list = [np.inf, thickness, 90, np.inf] #in nm
        R_list[i].append(coh_tmm('s', n_list, d_list, 0, lambda_vac)['R'])
    R_list[i] = np.array(R_list[i])
    contrast[i + 1] = (R_list[i]/R_list[i][0] - 1)
'''