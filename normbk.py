#%%
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import time


#%%
img = cv2.imread('E:/228 Picture/Jingxu/hBN/220124/01-24-2022-127.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img0 = img[100:, 200:1700]
plt.figure()
plt.imshow(img0)

img1 = img0

# %%
color = ('r','g','b')
plt.figure()
for i,col in enumerate(color):
    histr = cv2.calcHist([img1],[i],None,[256],[0,256])
    plt.plot(histr/np.max(histr),color = col)

#%%
fit1 = np.zeros_like(img1, dtype = float)
x = np.linspace(0, img1.shape[1] - 1, img1.shape[1])
for i in range(img1.shape[0]):
    for j in range(3):
        y = img1[i, :, j]/255
        index = reject_outliers(y)
        z = np.polyfit(x[index], y[index], 3)
        fit1[i, :, j] = z[0] * x**3 + z[1] * x**2 + z[2] * x + z[3]

fit2 = np.zeros_like(img1, dtype = float)
x = np.linspace(0, img1.shape[0] - 1, img1.shape[0])
for i in range(img1.shape[1]):
    for j in range(3):
        y = img1[:, i, j]/255
        index = reject_outliers(y)
        z = np.polyfit(x[index], y[index], 3)
        fit2[:, i, j] = z[0] * x**3 + z[1] * x**2 + z[2] * x + z[3]

fit = (fit1 + fit2) / 2
plt.figure()
plt.imshow(fit)

X = np.linspace(0, fit.shape[1] - 1, fit.shape[1])
Y = np.linspace(0, fit.shape[0] - 1, fit.shape[0])
X, Y = np.meshgrid(X, Y)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, fit[:, :, 0], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# %%
def reject_outliers(data, m = 0.2):
    d = np.abs(data - np.median(data))
    # mdev = np.median(d)
    # s = d/mdev
    s = d/np.median(data)
    index = s<m
    return index

def polyfit(x, y, index, poly = 4, threshold = 0.015):
    z1 = np.polyfit(x[index], y[index], poly)
    y1 = 0
    for i in range(poly + 1):
        y1 += z1[i] * x**(poly - i)
    d = np.abs(y - y1)
    s = d / y1
    
    index1 = s < threshold
    z2 = np.polyfit(x[index1], y[index1], poly)
    y2 = 0
    for i in range(poly + 1):
        y2 += z2[i] * x**(poly - i)
    return y2



#%% 
start = time.time()

img1 = cv2.resize(img0, (int(img0.shape[1]/1), int(img0.shape[0]/1)))

fit1 = np.zeros_like(img1, dtype = float)
x = np.linspace(0, img1.shape[1] - 1, img1.shape[1])
for i in range(img1.shape[0]):
    for j in range(3):
        y = img1[i, :, j]/255
        index = reject_outliers(y)
        fit1[i, :, j] = polyfit(x, y, index)

fit2 = np.zeros_like(img1, dtype = float)
x = np.linspace(0, img1.shape[0] - 1, img1.shape[0])
for i in range(img1.shape[1]):
    for j in range(3):
        y = img1[:, i, j]/255
        index = reject_outliers(y)
        fit2[:, i, j] = polyfit(x, y, index)

fit0 = (fit1 + fit2) / 2

print(time.time() - start)
plt.figure()
plt.imshow(fit0[:, :, 2])

# X = np.linspace(0, fit0.shape[1] - 1, fit0.shape[1])
# Y = np.linspace(0, fit0.shape[0] - 1, fit0.shape[0])
# X, Y = np.meshgrid(X, Y)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(X, Y, fit1[:, :, 0], cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# ax.set_title('fit1')

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(X, Y, fit2[:, :, 0], cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# ax.set_title('fit2')

# %%
# dst = cv2.blur(fit0, (5, 5))
# fit = cv2.resize(dst, (img0.shape[1], img0.shape[0]))

temp = np.float64(img0)
out = np.zeros_like(temp)
for i in range(3):
    out[:, :, i] = temp[:, :, i]/fit[:, :, i]*np.median(fit[:, :, i])
out = np.uint8(out)
plt.figure()
plt.imshow(out[:, :, 0])

# X = np.linspace(0, out.shape[1] - 1, out.shape[1])
# Y = np.linspace(0, out.shape[0] - 1, out.shape[0])
# X, Y = np.meshgrid(X, Y)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(X, Y, out[:, :, 2], cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)


# %%
lower_reso = cv2.pyrDown(img1)
lower_reso = cv2.pyrDown(lower_reso)

y = lower_reso[:, 0, 0]
x = np.linspace(0, len(y)-1, len(y))
index = reject_outliers(y, m  = 0.08)

poly = 3
threshold = 0.015
z1 = np.polyfit(x[index], y[index], poly)
y1 = 0
for i in range(poly + 1):
    y1 += z1[i] * x**(poly - i)
d = np.abs(y - y1)
s = d / y1

index1 = s < threshold
z2 = np.polyfit(x[index1], y[index1], poly)
y2 = 0
for i in range(poly + 1):
    y2 += z2[i] * x**(poly - i)

plt.figure()
plt.plot(x, y)
plt.plot(x[index], y[index])
plt.plot(x, y1)
plt.plot(x[index1], y[index1])
plt.plot(x, y2)
# %%
