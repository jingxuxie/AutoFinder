#%%
import cv2
import matplotlib.pyplot as plt
# %matplotlib qt
import numpy as np
from numba import jit
import time
import sys
sys.path.append("..")
import os
import faiss
from shutil import copyfile

#%%
@jit(nopython = True)
def background_divide(a, b, c):
    if len(a.shape) == 3:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                for k in range(a.shape[2]):
                    temp = round(int(a[i,j,k])/(b[i,j,k]/(c[k])+0.00001))
                    if temp >=255:
                        a[i,j,k] = 255
                    else:
                        a[i,j,k] = temp                    
    return a

#%%
def fine_segment(data, area_thresh = 200, variance_limit = np.array([6, 6, 6])):
    seg_done = []
    seg_further = []
    kmeans = faiss.Kmeans(d=3, k=2)
    kmeans.train(data.astype(np.float32))
    labels = kmeans.index.search(data.astype(np.float32), 1)[1].squeeze()
    for i in range(2):
        seg = data[labels == i]
        if len(seg) < area_thresh:
            continue
        variance = np.std(seg, axis = 0)
        if (variance < variance_limit).all():
            seg_done.append(seg)
        else:
            seg_further.append(seg)
    return seg_done, seg_further


def calculate_contrast(sample_color, bk_color):
    '''
    color: np.array in the order of blue, green, red (BGR)
    '''
    contrast_range = np.array([[-0.02, 0.15], [0.1, 0.28], [0.17, 0.28]])
    contrast = (bk_color - sample_color) / bk_color
    for i in range(3):
        if not contrast_range[i, 0] <= contrast[i] <= contrast_range[i, 1]:
            return False, contrast
    return True, contrast


def get_local_bk(data, bk_color):
    bk_color_local = np.zeros(3)
    hist = [0, 0, 0]
    delta = 5
    for i in range(3):
        hist[i] = cv2.calcHist([data[:, :, i]], [0], None, [256], [0,255])
        hist[i] = hist[i][bk_color[i] - delta: bk_color[i] + delta + 1]
        bk_color_local[i] = np.argmax(hist[i]) + bk_color[i] - delta
    return bk_color_local



#%%
def find_segment_list(X, area_thresh = 200, variance_limit = np.array([6, 6, 6])):
    variance = np.std(X, axis = 0)
    segments = []
    seg_further_list = []
    if (variance < variance_limit).all():
        segments.append(X)
    else:
        seg_done, seg_further = fine_segment(X, area_thresh, variance_limit)
        segments += seg_done
        seg_further_list += seg_further
        index = 0
        while index < len(seg_further_list):
            seg_done, seg_further = fine_segment(seg_further_list[index], area_thresh, variance_limit)
            segments += seg_done
            seg_further_list += seg_further
            index += 1
    # print(len(segments), [np.mean(segments[i], axis = 0) for i in range(len(segments))])
    return segments



def careful_look(data, segments, local_bk_color, area_thresh = 200):
    real_flake_list = []
    for seg in segments:
        mean = np.mean(seg, axis = 0)
        
        ret, contrast = calculate_contrast(mean, local_bk_color)
        # print(contrast)
        if not ret:
            continue
        
        std = np.std(seg, axis = 0)
        lb = mean - 1.2*std
        ub = mean + 1.2*std
        img_binary = cv2.inRange(data, lb, ub)

        kernel_close = np.ones((3,3),np.uint8)
        img_close = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = np.ones((3,3),np.uint8)
        img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(img_open, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            # print(area)
            if area > area_thresh:
                mask = np.zeros(img_open.shape[:2], np.uint8)
                cv2.drawContours(mask, [contours[i]], -1, 255, -1)
                piece = data[mask==255]
                variance = np.std(piece, axis = 0)
                # if not (variance < 5).all():
                #     continue
                mean = np.median(piece, axis = 0)
                ret, contrast = calculate_contrast(mean, local_bk_color)
                if ret:
                    real_flake_list.append(contrast)
                    break
        # print(contrast)
        # plt.figure()
        # plt.imshow(img_open, cmap = 'gray')
        

    if len(real_flake_list) > 0:
        return True, real_flake_list
    return False, []




#%%
def layer_search_wse2(filename, area_thresh = 50):
    isLayer = False

    bk = cv2.imread('F:/Temp/gr/10x_new.png')
    height, width, _ = bk.shape

    crop_h = [0, 1]
    crop_w = [0.15, 0.75]

    roi_w = [int(width * crop_w[0]), int(width * crop_w[1])]
    roi_h = [int(height * crop_h[0]), int(height * crop_h[1])]

    bk = bk[roi_h[0]: roi_h[1], roi_w[0]: roi_w[1]]
    median = np.median(bk, axis = (0, 1))

    # filename = 'F:/Temp/ws2/6-25-3/small_0/2023-06-25--13-29-10-706686.jpg'
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
            return False, []

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
    threshold_low = int(bk_color_y * 0.7)
    threshold = int(bk_color_y * 0.9)

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
        segments = find_segment_list(temp, area_thresh)
        ret, contrast_list = careful_look(img_cnt_large_cut, segments, local_bk_color, area_thresh)
        
        if ret:
            isLayer = True
            cv2.rectangle(img_for_draw, (x-w, y-h), (x+3*w, y+3*h), (0, 0, 255), 2)

    return isLayer, img_for_draw

#%%
def test_run():
    filepath = 'F:/Temp/wse2/6-26-2/small_1'
    pathDir =  os.listdir(filepath)
    # outpath = 'F:/Temp/gr/6-24/temp'
    resultpath = 'F:/Temp/wse2/6-26-2/results_1_50'
    if not os.path.isdir(resultpath):
        os.makedirs(resultpath)
    finished_count = 0
    for finished_count in range(len(pathDir) - 2):
        print(finished_count, pathDir[finished_count])
        # output_name = outpath+'/'+pathDir[finished_count]
        input_name = filepath+'/'+pathDir[finished_count]

        result_name = resultpath+'/'+pathDir[finished_count]
        ret, img_out = layer_search_wse2(input_name)
        
        if ret:
            print(ret)
            cv2.imwrite(result_name, img_out)
    

#%%
if __name__ == '__main__':
    test_run()

# %%
