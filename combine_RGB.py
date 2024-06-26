#%%
import os
import cv2
import numpy as np
from autofinder.auxiliary_func import generate_positions
import cupy, cupyx
from cupyx.scipy.signal import fftconvolve
import time


#%%
img1 = cv2.imread('F:/Temp/bn_new/0810-5/raw/2023-08-10--20-54-56-57G000.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('F:/Temp/bn_new/0810-5/raw/2023-08-10--20-54-57-09B000.png', cv2.IMREAD_GRAYSCALE)


def get_drift(img1, img2):
    w = 1500
    h = 1500
    center_l = 0
    center_h = 1500
        
    image1 = cv2.resize(img1, (w, h)).astype(np.float32)
    image2 = cv2.resize(img2, (w, h)).astype(np.float32)

    image1 -= np.mean(image1)
    image1 /= np.std(image1)
    image2 -= np.mean(image2)
    image1 /= np.std(image1)

    image1 = cupy.array(image1)
    image2 = cupy.array(image2)

    out = fftconvolve(image1, image2[::-1,::-1], mode='same')
    index = cupy.unravel_index(cupy.argmax(out[center_l:center_h, 
                                               center_l:center_h]), 
                                               (1500, 1500))

    y_shift = (750 - int(index[0])) * 2 
    x_shift = (750 - int(index[1])) * 2

    print(y_shift, x_shift)
    return [y_shift, x_shift]


# %%
def shift_image_back(img1, img2):
    y_shift, x_shift = get_drift(img1, img2)
    img_out = np.zeros_like(img2, dtype = np.uint8)

    # y_shift = (50 - drift[0]) * 2 
    # x_shift = (50 - drift[1]) * 2

    if y_shift > 0:
        if x_shift > 0:
            img_out[:-y_shift, :-x_shift] = img2[y_shift:, x_shift:]
        elif x_shift == 0:
            img_out[:-y_shift, :] = img2[y_shift:, :]
        else:
            img_out[:-y_shift, -x_shift:] = img2[y_shift:, :x_shift]
    elif y_shift == 0:
        if x_shift > 0:
            img_out[:, :-x_shift] = img2[:, x_shift:]
        elif x_shift == 0:
            img_out[:, :] = img2[:, :]
        else:
            img_out[:, -x_shift:] = img2[:, :x_shift]
    else:
        if x_shift > 0:
            img_out[-y_shift:, :-x_shift] = img2[:y_shift, x_shift:]
        elif x_shift == 0:
            img_out[-y_shift:, :] = img2[:y_shift, :]
        else:
            img_out[-y_shift:, -x_shift:] = img2[:y_shift, :x_shift]
    return img_out

#%%

start = time.time()

input_folder = 'F:/Temp/ws2_new/0815/small_2/'
files = os.listdir(input_folder)
_, file_list, position_list = generate_positions(input_folder)

output_folder = 'F:/Temp/ws2_new/0815/color_shift_2/'
position_filename = output_folder + 'position.txt'

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

count = 0
img_new = np.zeros((3000, 3000, 3), dtype = np.uint8)
for i in range(len(files) - 1):
    filename = files[i]
    file = input_folder + filename
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    if count == 0:
        img_new[:, :, 1] = img
    elif count == 1:
        img_new[:, :, 2] = shift_image_back(img_new[:, :, 1], img)
    else:
        img_new[:, :, 0] = shift_image_back(img_new[:, :, 1], img)
    
    if count == 2:
        count = -1
        cv2.imwrite(output_folder + filename, img_new)

        with open(position_filename, 'a') as file:
            pos = list(position_list[i]) + [0.0]
            file.write(filename + ' ' + ' '.join(map(str, pos)) + '\n')

    count += 1

print(time.time() - start)

# plt.figure()
# plt.imshow(img_out)
# %%
