#%%
import cv2
import matplotlib.pyplot as plt
# %matplotlib qt
import numpy as np
import json
import time
import sys
sys.path.append("..")
import os
from shutil import copyfile
from autofinder.find_layers.predict_wse2 import predict_wse2
from autofinder.auxiliary_func import background_divide, get_background
from autofinder.find_layers.find_layers_func import get_local_bk,\
find_segment_list, careful_look, put_Text
from autofinder.auxiliary_func import generate_positions
import scipy


#%%
def layer_search_wse2(filename, background, area_thresh = 200, thickness_range = [0.3, 2.3],
                     roi = [0, 1, 0, 1]):
    isLayer = False
    contrast_list = []
    flake_position_list = []
    predictor = predict_wse2

    bk = background
    height, width, _ = bk.shape

    crop_w = roi[:2]
    crop_h = roi[2:]

    roi_w = [int(width * crop_w[0]), int(width * crop_w[1])]
    roi_h = [int(height * crop_h[0]), int(height * crop_h[1])]

    bk = bk[roi_h[0]: roi_h[1], roi_w[0]: roi_w[1]]
    median = np.median(bk, axis = (0, 1))


    # filename = 'F:/Temp/wse2_new/0812/color_shift_7/2023-08-12--18-35-50-36B000.png'
    img = cv2.imread(filename)
    img = img[roi_h[0]: roi_h[1], roi_w[0]: roi_w[1]]

    img = background_divide(img, bk, median)
    img_for_draw = img.copy()
    # plt.figure()
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    hist = [[] for i in range(3)]
    bk_color = np.zeros(3, dtype = np.int32)
    for i in range(3):
        hist[i] = cv2.calcHist([img[:, :, i]], [0], None, [256], [0,255]).squeeze()
        hist[i] = hist[i][65: 130]
        index = np.argmax(hist[i])
        # bk_color[i] = index + 70
        if hist[i][index] < 1e5:
            return False, [], [], []
    index = get_real_bk_color(hist)
    bk_color = index + 65

    print(bk_color)
    

    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # hist_y = cv2.calcHist([img_gray], [0], None, [256], [0,255])
    # hist_y = hist_y[70: 130]
    # bk_color_y = np.argmax(hist_y) + 70
    
    # remove edge, maybe a better way to solve this is to find edge
    img_binary = cv2.inRange(img[:, :, 0], 0, 40)
    kernel_open = np.ones((20,20),np.uint8)
    img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel_open)
    kernel_dilation = np.ones((35, 35), np.uint8)
    img_dilation = cv2.dilate(img_open, kernel_dilation, iterations=1)

    contours, _ = cv2.findContours(img_dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 5e5:
            mask = np.zeros(img.shape[:2], np.uint8)
            cv2.drawContours(mask, [contours[i]], -1, 255, -1)
            mask_dilation = cv2.dilate(mask, np.ones((500, 500), np.uint8))
            img[mask_dilation == 255] = 0

    img[img_dilation==255] = 0
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # plt.figure()
    # plt.imshow(img_gray, cmap='gray')

    #%%
    threshold_lo = [int(bk_color[0] * 0.4), int(bk_color[1] * 0.25), int(bk_color[2] * 0.25)]
    threshold_hi = [int(bk_color[0] * 0.9), int(bk_color[1] * 0.9), int(bk_color[2] * 0.8)]

    img_binary = cv2.inRange(img, np.array(threshold_lo), np.array(threshold_hi))


    kernel_open = np.ones((7,7),np.uint8)
    img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel_open)

    kernel_close = np.ones((7,7),np.uint8)
    img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel_close)

    #%%
    contours, _ = cv2.findContours(img_close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cnt_large_ensemble = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > area_thresh:
            cnt_large_ensemble.append(contours[i])
        
    # draw all the contours
    # image = cv2.drawContours(img.copy(), cnt_large_ensemble, -1, (0,0,255), 3)
    # plt.figure()
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    #%%
    for cnt_large in cnt_large_ensemble:
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [cnt_large], -1, 255, -1)
        img_cnt_large_cut = cv2.bitwise_and(img, img, mask = mask)
        x,y,w,h = cv2.boundingRect(cnt_large)
        img_rec = img[y:y+h, x:x+w]
        if len(img_rec[img_rec[:, :, 2] == 0]) > 10:
            # print('this is around a dust')
            continue

        # plt.figure()
        # plt.imshow(cv2.cvtColor(img_rec, cv2.COLOR_BGR2RGB))

        img_cnt_large_cut = img_cnt_large_cut[y : y + h, x : x + w]
        img_local = img[max(0, y - h) : min(img.shape[0], y + 3*h), 
                        max(0, x - w) : min(img.shape[1], x + 3*w)]
        local_bk_color = get_local_bk(img_local, bk_color)
        temp = img[mask == 255]
        segments = find_segment_list(temp, area_thresh, variance_limit = np.array([5, 5, 5]))
        ret, contrast_list_local, thickness_list = careful_look(img_cnt_large_cut, segments, 
                                                                local_bk_color, predictor, area_thresh, 
                                                                thickness_range)
        # print(ret, thickness_list)
        if ret:
            isLayer = True
            cv2.rectangle(img_for_draw, (x-w, y-h), (x+2*w, y+2*h), (0, 0, 255), 2)
            put_Text(img_for_draw, thickness_list, (x, y, w, h))
            contrast_list.append(contrast_list_local)
            flake_position_list.append([x, y, w, h])


    return isLayer, img_for_draw, contrast_list, flake_position_list


#%%
def get_real_bk_color(hist):
    index = np.zeros(3, dtype = int)
    for i in range(3):
        data = hist[i]
        peaks, _ = scipy.signal.find_peaks(data)
        indices = sorted(peaks, key=lambda x: data[x], reverse=True)
        index[i] = indices[0]
        if len(indices) >= 2:
            if data[indices[1]] > np.max(data) / 7:
                if i == 0 and indices[1] < indices[0]:
                    index[i] = indices[1]
                elif i == 2 and indices[1] > indices[0]:
                    index[i] = indices[1]
    return index
            



#%%
def test_run(background):
    filepath = 'F:/Temp/wse2_new/0813/color_shift_8'
    _, file_list, _ = generate_positions(filepath)
    resultpath = 'F:/Temp/wse2_new/0813/results_8'
    contrast_json = {'items': []}
    if not os.path.isdir(resultpath):
        os.makedirs(resultpath)
    for finished_count in range(len(file_list)):
        filename = file_list[finished_count]
        print(finished_count, filename)
        input_name = filepath+'/'+filename
        result_name = resultpath+'/'+filename
        ret, img_out, contrast_list, flake_position_list = layer_search_wse2(input_name, background)
        
        if ret:
            print(ret)
            cv2.imwrite(result_name, img_out)

            item = {'filename': filename,
                    'contrast_list': contrast_list,
                    'flake_position_list': flake_position_list}
            contrast_json['items'].append(item)
            a = json.dumps(contrast_json)
            with open(resultpath + '/flakes_info.json', 'w') as f:
                f.write(a)
    

#%%
if __name__ == '__main__':
    bk = get_background('F:/Temp/wse2_new/0813/color_shift_8', 1500, 1500)
    bk = cv2.resize(bk, (3000, 3000))
    # bk = cv2.imread('F:/Temp/wse2_new/0812/color_shift_3/bk.png')
    test_run(bk)

# %%
