#%%
import cv2
import matplotlib.pyplot as plt
# %matplotlib qt
import numpy as np
import time
import sys
sys.path.append("..")
import os
from shutil import copyfile
from autofinder.find_layers.predict_ws2 import predict_ws2
from autofinder.auxiliary_func import background_divide, get_background
from autofinder.find_layers.find_layers_func import get_local_bk,\
find_segment_list, careful_look, put_Text


#%%
def layer_search_ws2(filename, background, area_thresh = 50, thickness_range = [0, 2],
                     roi = [0.15, 0.75, 0, 1]):
    isLayer = False
    contrast_list = []
    flake_position_list = []
    predictor = predict_ws2

    bk = background
    height, width, _ = bk.shape

    crop_w = roi[:2]
    crop_h = roi[2:]

    roi_w = [int(width * crop_w[0]), int(width * crop_w[1])]
    roi_h = [int(height * crop_h[0]), int(height * crop_h[1])]

    bk = bk[roi_h[0]: roi_h[1], roi_w[0]: roi_w[1]]
    median = np.median(bk, axis = (0, 1))

    # filename = 'F:/Temp/ws2/7-6-1/small_0/2023-07-06--12-28-04-47000.jpg'
    img = cv2.imread(filename)
    img = img[roi_h[0]: roi_h[1], roi_w[0]: roi_w[1]]

    img = background_divide(img, bk, median)
    img_for_draw = img.copy()
    # plt.figure()
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    hist = [[] for i in range(3)]
    bk_color = np.zeros(3, dtype = np.int32)
    for i in range(3):
        hist[i] = cv2.calcHist([img[:, :, i]], [0], None, [256], [0,255])
        hist[i] = hist[i][90: 200]
        index = np.argmax(hist[i])
        bk_color[i] = index + 90
        if hist[i][index] < 1e5:
            return False, [], [], []

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist_y = cv2.calcHist([img_gray], [0], None, [256], [0,255])
    hist_y = hist_y[90: 200]
    bk_color_y = np.argmax(hist_y) + 90
    
    # remove edge, maybe a better way to solve this is to find edge
    img_binary = cv2.inRange(img[:, :, 0], 0, 90)
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
            mask_dilation = cv2.dilate(mask, np.ones((200, 200), np.uint8))
            img[mask_dilation == 255] = 0

    img[img_dilation==255] = 0
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # plt.figure()
    # plt.imshow(img_gray, cmap='gray')

    #%%
    threshold_low = int(bk_color_y*0.8)
    threshold = bk_color_y - 1

    img_binary = cv2.inRange(img_gray, np.array(threshold_low), np.array(threshold))

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
    image = cv2.drawContours(img.copy(), cnt_large_ensemble, -1, (0,0,255), 3)
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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
        img_local = img[max(0, y - h) : min(img.shape[0], y + 2*h), 
                        max(0, x - w) : min(img.shape[1], x + 2*w)]
        local_bk_color = get_local_bk(img_local, bk_color)
        temp = img[mask == 255]
        segments = find_segment_list(temp, area_thresh, variance_limit=np.array([5, 5, 7]))
        ret, contrast_list_local, thickness_list = careful_look(img_cnt_large_cut, segments, 
                                                                local_bk_color, predictor, area_thresh,
                                                                thickness_range)
        
        if ret:
            isLayer = True
            cv2.rectangle(img_for_draw, (x-w, y-h), (x+2*w, y+2*h), (0, 0, 255), 2)
            put_Text(img_for_draw, thickness_list, (x, y, w, h))
            contrast_list.append(contrast_list_local)
            flake_position_list.append([x, y, w, h])

    return isLayer, img_for_draw, contrast_list, flake_position_list

#%%
def test_run():
    filepath = 'F:/Temp/ws2/6-26-1/small_2'
    pathDir =  os.listdir(filepath)
    # outpath = 'F:/Temp/gr/6-24/temp'
    resultpath = 'F:/Temp/ws2/6-26-1/results_2_50'
    if not os.path.isdir(resultpath):
        os.makedirs(resultpath)
    finished_count = 0
    for finished_count in range(len(pathDir) - 2):
        print(finished_count, pathDir[finished_count])
        # output_name = outpath+'/'+pathDir[finished_count]
        input_name = filepath+'/'+pathDir[finished_count]

        result_name = resultpath+'/'+pathDir[finished_count]
        ret, img_out = layer_search_ws2(input_name)
        
        if ret:
            print(ret)
            cv2.imwrite(result_name, img_out)
    

#%%
if __name__ == '__main__':
    test_run()


# %%
