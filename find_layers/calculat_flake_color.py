#%%
import numpy as np
import scipy

#%%
from tmm.tmm_core import coh_tmm

from numpy import pi, linspace, inf, array
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


material_nk_data = array([[400, 5.57 + 0.387j],
                          [450, 4.67 + 0.145j],
                          [500, 4.30 + 7.28e-2j],
                          [550, 4.08 + 4.06e-2j],
                          [600, 3.95 + 2.57e-2j],
                          [650, 3.85 + 1.64e-2j],
                          [700, 3.78 + 1.26e-2j],
                          [750, 3.71 + 0.8e-2j],
                          [800, 3.67 + 0.6e-2j],
                          [1000, 3.6 + 0.1e-2j]])
data = scipy.io.loadmat('E:/Desktop2023.1.17/AutoFinder/matlab bn/refractiveindex.mat')
wavelength = data['lambda'].squeeze()
nk_si = data['n_Si'].squeeze()
nk_sio2 = data['n_SiO2'].squeeze()
nk_bn = data['n_BN'].squeeze()
nk_gr = data['n_Gr'].squeeze()
nk_ws2 = np.load('E:/Desktop2023.1.17/AutoFinder/nk_WS2.npy')
nk_wse2 = np.load('E:/Desktop2023.1.17/AutoFinder/nk_WSe2.npy')
nk_mose2 = np.load('E:/Desktop2023.1.17/AutoFinder/nk_MoSe2.npy')

nk_fn_si = interp1d(wavelength, nk_si, )
nk_fn_sio2 = interp1d(wavelength, nk_sio2, kind='linear')
nk_fn_bn = interp1d(wavelength, nk_bn, )
nk_fn_gr = interp1d(wavelength, nk_gr, )
nk_fn_ws2 = interp1d(nk_ws2[0], nk_ws2[1])
nk_fn_wse2 = interp1d(nk_wse2[0], nk_wse2[1])
nk_fn_mose2 = interp1d(nk_mose2[0], nk_mose2[1])

#%%
thicknesslist = [1000000]
plt.figure()
for thickness in thicknesslist:
    d_list = [inf, thickness, inf] #in nm
    lambda_list = linspace(400, 800, 400) #in nm
    T_list = []
    for lambda_vac in lambda_list:
        n_list = [1, nk_fn_sio2(lambda_vac)[0], nk_fn_si(lambda_vac)[0]]
        T_list.append(coh_tmm('p', n_list, d_list, 0.3, lambda_vac)['R'])

    plt.plot(lambda_list, T_list)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Fraction of power reflected')

#%%
thicknesslist = np.linspace(0, 5, 501)
lambda_list = [452, 537, 633]
color_list = ['b', 'g', 'r']
plt.figure()
for i in range(len(lambda_list[:3])):
    lambda_vac = lambda_list[i]
    n_list = [1, nk_fn_mose2(lambda_vac), nk_fn_sio2(lambda_vac), nk_fn_si(lambda_vac)]
    R_list = []
    for thickness in thicknesslist:
        d_list = [inf, thickness, 90, inf] #in nm
        R_list.append(coh_tmm('p', n_list, d_list, 0, lambda_vac)['R'])
    R_list = np.array(R_list)
    contrast = R_list/R_list[0] - 1
    plt.plot(thicknesslist, contrast, color = color_list[i])

# plt.xlim([0, 30])
# plt.ylim([-0.5, 6])

# plt.legend({'1','2','3'})
# plt.ylim([0, 0.5])


#%%
flake_thickness_list = np.linspace(0, 15, 151)
sio2_thicknesslist = np.linspace(80, 100, 21)
lambda_list = np.linspace(430, 670, 241)
theta_list = np.linspace(0, 0.3046, 31)
polarization_list = ['s', 'p']

out = np.zeros((151, 241, 31, 2, 21))
color_list = ['r', 'g', 'b']
plt.figure()
for m in range(len(flake_thickness_list)):
    flake_thickness = flake_thickness_list[m]
    print(m)
    for i in range(len(lambda_list)):
        lambda_vac = lambda_list[i]
        n_list = [1,  nk_fn_gr(lambda_vac)[0], nk_fn_sio2(lambda_vac)[0], nk_fn_si(lambda_vac)[0]]
        for j in range(len(theta_list)):
            theta = theta_list[j]
            for k in range(2):
                polarization = polarization_list[k]
                R_list = []
                for sio2_thickness in sio2_thicknesslist:
                    d_list = [inf, flake_thickness, sio2_thickness, inf] #in nm
                    R_list.append(coh_tmm(polarization, n_list, d_list, theta, lambda_vac)['R'])
                R_list = np.array(R_list)
                out[m, i, j, k] = R_list


#%%
# integrate over angle
data = np.load('E:/Desktop2023.1.17/AutoFinder/gr_0_15_151_420_670_251.npy')
data_after_theta = np.zeros([data.shape[0], data.shape[1], data.shape[3], data.shape[-1]])
theta_list = np.linspace(0, 0.3046, 31)
total = 0
for i in range(30):
    theta = (theta_list[i + 1] + theta_list[i]) / 2
    delta_theta = theta_list[i + 1] - theta_list[i]
    total += np.sin(theta) * delta_theta

    data_at_theta = (data[:, :, i + 1] + data[:, :, i])/2
    data_after_theta += np.sin(theta) * data_at_theta * delta_theta

data_after_theta /= total

data_after_polar = np.zeros([data.shape[0], data.shape[1], data.shape[-1]])
data_after_polar = (data_after_theta[:, :, 0] + data_after_theta[:, :, 1])/2

data_after_polar = data[:, :, 0, 0, :] #no angle integral

#%% integrate over wavelength intensity
intensity_b = np.load('E:/Desktop2023.1.17/AutoFinder/intensity_b.npy')
intensity_g = np.load('E:/Desktop2023.1.17/AutoFinder/intensity_g.npy')
intensity_r = np.load('E:/Desktop2023.1.17/AutoFinder/intensity_r.npy')
intensity_b_fn = interp1d(intensity_b[0], intensity_b[1])
intensity_g_fn = interp1d(intensity_g[0], intensity_g[1])
intensity_r_fn = interp1d(intensity_r[0], intensity_r[1])

data_after_wavelength = np.zeros((3, data.shape[0], data.shape[-1]))
wavelength_list = np.linspace(420, 670, 251)
total = 0
for i in range(80):
    wavelength = (wavelength_list[i + 1] + wavelength_list[i])/2
    delta_wavelength = wavelength_list[i + 1] - wavelength_list[i]
    total += intensity_b_fn(wavelength) * delta_wavelength
    
    data_at_wavelength = (data_after_polar[:, i + 1] + data_after_polar[:, i]) / 2
    data_after_wavelength[0] += data_at_wavelength * intensity_b_fn(wavelength) * delta_wavelength
data_after_wavelength[0] /= total

total = 0
for i in range(90, 170):
    wavelength = (wavelength_list[i + 1] + wavelength_list[i])/2
    delta_wavelength = wavelength_list[i + 1] - wavelength_list[i]
    total += intensity_g_fn(wavelength) * delta_wavelength
    
    data_at_wavelength = (data_after_polar[:, i + 1] + data_after_polar[:, i]) / 2
    data_after_wavelength[1] += data_at_wavelength * intensity_g_fn(wavelength) * delta_wavelength
data_after_wavelength[1] /= total

total = 0
for i in range(190, 230):
    wavelength = (wavelength_list[i + 1] + wavelength_list[i])/2
    delta_wavelength = wavelength_list[i + 1] - wavelength_list[i]
    total += intensity_r_fn(wavelength) * delta_wavelength
    
    data_at_wavelength = (data_after_polar[:, i + 1] + data_after_polar[:, i]) / 2
    data_after_wavelength[2] += data_at_wavelength * intensity_r_fn(wavelength) * delta_wavelength
data_after_wavelength[2] /= total

plt.figure()
plt.plot(np.linspace(0, 15, 151), data_after_wavelength[0, :, 10] / data_after_wavelength[0, 0, 10] - 1, color = 'b')
plt.plot(np.linspace(0, 15, 151), data_after_wavelength[1, :, 10] / data_after_wavelength[1, 0, 10] - 1, color = 'g')
plt.plot(np.linspace(0, 15, 151), data_after_wavelength[2, :, 10] / data_after_wavelength[2, 0, 10] - 1, color = 'r')


#%%  
file = 'E:/Desktop2023.1.17/AutoFinder/b-200_1.txt'
with open(file) as f:
    lines = f.readlines()
intensity_b = np.zeros((2, 1024))
for i in range(1024):
    line = lines[i].split()
    intensity_b[0, i] = float(line[0])
    intensity_b[1, i] = float(line[1])
intensity_b[0] -= 200
intensity_b[1] -= 630
intensity_b[1] /= np.max(intensity_b[1])
plt.figure()
plt.plot(intensity_b[0], intensity_b[1], color = 'b')
np.save('E:/Desktop2023.1.17/AutoFinder/intensity_b.npy', intensity_b)

file = 'E:/Desktop2023.1.17/AutoFinder/g-200_1.txt'
with open(file) as f:
    lines = f.readlines()
intensity_g = np.zeros((2, 1024))
for i in range(1024):
    line = lines[i].split()
    intensity_g[0, i] = float(line[0])
    intensity_g[1, i] = float(line[1])
intensity_g[0] -= 200
intensity_g[1] -= 630
intensity_g[1] /= np.max(intensity_g[1])
plt.plot(intensity_g[0], intensity_g[1], color = 'g')
np.save('E:/Desktop2023.1.17/AutoFinder/intensity_g.npy', intensity_g)

file = 'E:/Desktop2023.1.17/AutoFinder/r-100_1.txt'
with open(file) as f:
    lines = f.readlines()
intensity_r = np.zeros((2, 1024))
for i in range(1024):
    line = lines[i].split()
    intensity_r[0, i] = float(line[0])
    intensity_r[1, i] = float(line[1])
intensity_r[0] -= 100
intensity_r[1] -= 630
intensity_r[1] /= np.max(intensity_r[1])
plt.plot(intensity_r[0], intensity_r[1], color = 'r')
np.save('E:/Desktop2023.1.17/AutoFinder/intensity_r.npy', intensity_r)


#%% thorlab silver mirror reflection
thorlab_wave = np.linspace(420, 700, 29)
reflection = [93.09852, 94.34188, 95.12392, 95.73861, 96.07818, 96.27313, 96.43713,
96.52764, 96.55209, 96.50419, 96.42279, 96.37185, 96.33421, 96.34123, 96.21805,
96.07214, 95.904, 95.85272, 95.75384, 95.73035, 95.72144, 95.7127, 95.71157,
95.72349, 95.68295, 95.66181, 95.58834, 95.57215, 95.39356]
cali_reflection = np.zeros(3)
total = 0
for i in range(9):
    wavelength = (thorlab_wave[i + 1] + thorlab_wave[i])/2
    delta_wavelength = 10
    total += intensity_b_fn(wavelength) * delta_wavelength
    
    reflection_at_wavelength = (reflection[i + 1] + reflection[i]) / 2
    cali_reflection[0] += reflection_at_wavelength * intensity_b_fn(wavelength) * delta_wavelength
cali_reflection[0] /= total

total = 0
for i in range(8, 18):
    wavelength = (thorlab_wave[i + 1] + thorlab_wave[i])/2
    delta_wavelength = 10
    total += intensity_g_fn(wavelength) * delta_wavelength
    
    reflection_at_wavelength = (reflection[i + 1] + reflection[i]) / 2
    cali_reflection[1] += reflection_at_wavelength * intensity_g_fn(wavelength) * delta_wavelength
cali_reflection[1] /= total

total = 0
for i in range(18, 25):
    wavelength = (thorlab_wave[i + 1] + thorlab_wave[i])/2
    delta_wavelength = 10
    total += intensity_r_fn(wavelength) * delta_wavelength
    
    reflection_at_wavelength = (reflection[i + 1] + reflection[i]) / 2
    cali_reflection[2] += reflection_at_wavelength * intensity_r_fn(wavelength) * delta_wavelength
cali_reflection[2] /= total

#%%
def nk_ws2(wavelength):
    out = np.sqrt(LorentzPermitivity(wavelength, 17, 2.8, 2.03, 5.5e-3))
    return out

def nk_wse2(wavelength):
    out = np.sqrt(LorentzPermitivity(wavelength, 17, 2.9, 1.735, 5e-3))
    return out

def nk_mose2(wavelength):
    out = np.sqrt(LorentzPermitivity(wavelength, 17, 2.8, 1.65, 5.5e-3))
    return out

def LorentzPermitivity(wavelength, eps0, A, E0, gamme):
    E = 1239.847 / wavelength
    eps = eps0 + A / (E0**2 - E**2 - 1j * E * gamme)
    return eps
#%%
def refractiveindex(mater, wavelength):
    folder = 'E:/Desktop2023.1.17/AutoFinder/matlab bn/'
    scipy.io.loadmat(folder + mater + '.mat')

def FresnelReflection(n1, n2):
    return (n1 - n2) / (n1 + n2)

def SlabTotalReflection(r1, r2, n, d, wavelength):
    r01 = r1
    t01 = 1 + r01
    t10 = 1 - r01
    r12 = r2
    delta = 2 * np.pi * n * d / wavelength
    R = r01 + t01 * t10 * r12 * np.exp(2j * delta) / (1 + r01 * r12 * np.exp(2j * delta))
    return R

def TotalReflection(layern, layerd, wavelenth):
    '''
    layern: refractive index of each layer
    layerd: layer thickness
    '''
    N = len(layern)
    ii = N - 2
    n0 = layern
# %%
