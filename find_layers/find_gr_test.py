#%%
import cv2
import matplotlib.pyplot as plt
%matplotlib qt
import numpy as np
from numba import jit
import time
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
from sklearn.cluster import KMeans
# import os
# os.environ["OMP_NUM_THREADS"] = '1'
import faiss
from autofinder.find_layers.predict_gr import predict_gr
#%%
predict_gr([0.05, 0.05, 0.05])
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

@jit(nopython = True)
def region_merging(a, label, c):
    label[0, 0] = 1
    count = 1
    for i in range(a.shape[0] - 1):
        for j in range(a.shape[1] - 1):
            if label[i, j + 1] == 0:
                if abs(a[i, j + 1, 0] - a[i, j, 0]) < c:
                    label[i, j + 1] = label[i, j]
                else:
                    count += 1
                    label[i, j + 1] = count
            if abs(a[i + 1, j, 0] - a[i, j, 0]) < c:
                label[i + 1, j] = label[i, j]
    return label


#%%
bk = cv2.imread('F:/Temp/gr/10x.png')
height, width, _ = bk.shape

crop_h = [0, 1]
crop_w = [0.15, 0.75]

roi_w = [int(width * crop_w[0]), int(width * crop_w[1])]
roi_h = [int(height * crop_h[0]), int(height * crop_h[1])]

bk = bk[roi_h[0]: roi_h[1], roi_w[0]: roi_w[1]]
median = np.median(bk, axis = (0, 1))

filename = 'F:/Temp/gr/small_0/2023-06-20--15-47-02-329032.jpg'
img = cv2.imread(filename)
img = img[roi_h[0]: roi_h[1], roi_w[0]: roi_w[1]]


img = background_divide(img, bk, median)
plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# b,g,r = cv2.split(img)
# hist_b = cv2.calcHist([b], [0], None, [256], [0,255])
# hist_g = cv2.calcHist([g], [0], None, [256], [0,255])
# hist_r = cv2.calcHist([r], [0], None, [256], [0,255])

hist = [[] for i in range(3)]
bk_color = np.zeros(3, dtype = np.int32)
for i in range(3):
    hist[i] = cv2.calcHist([img[:, :, i]], [0], None, [256], [0,255])
    hist[i] = hist[i][90: 200]
    bk_color[i] = np.argmax(hist[i]) + 90

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist_y = cv2.calcHist([img_gray], [0], None, [256], [0,255])
hist_y = hist_y[90: 200]
bk_color_y = np.argmax(hist_y) + 90


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

plt.figure()
plt.imshow(img_gray)

#%%
threshold_lo = [int(bk_color[0] * 0.85), int(bk_color[1] * 0.2), int(bk_color[2] * 0.2)]
threshold_hi = [int(bk_color[0] * 1.25), int(bk_color[1] * 0.98), int(bk_color[2] * 0.98)]

img_binary = cv2.inRange(img, np.array(threshold_lo), np.array(threshold_hi))

kernel_open = np.ones((7,7),np.uint8)
img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel_open)

kernel_close = np.ones((7,7),np.uint8)
img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel_close)

# plt.figure()
# plt.plot(hist_y)
plt.figure()
# plt.imshow(img_gray)
# plt.imshow(img_binary, cmap = 'gray')
plt.imshow(img_close, cmap = 'gray')

#%%
contours, _ = cv2.findContours(img_close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cnt_large_ensemble = []
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area > 400:
        cnt_large_ensemble.append(contours[i])
    
# draw all the contours
image = cv2.drawContours(img.copy(), cnt_large_ensemble, -1, (0,0,255), 3)

plt.figure()
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#%%
for cnt_large in cnt_large_ensemble[:4]:
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [cnt_large], -1, 255, -1)
    img_cnt_large_cut = cv2.bitwise_and(img, img, mask = mask)
    x,y,w,h = cv2.boundingRect(cnt_large)
    img_cnt_large_cut = img_cnt_large_cut[y : y + h, x : x + w]
    img_local = img[max(0, y - h) : min(img.shape[0], y + 3*h), 
                    max(0, x - w) : min(img.shape[1], x + 3*w)]
    img_rec = img[y:y+h, x:x+w]
    if len(img_rec[img_rec[:, :, 0] == 0]) > 10:
        print('this is around a dust')
    local_bk_color = get_local_bk(img_local)
    temp = img[mask == 255]


plt.figure()
plt.imshow(cv2.cvtColor(img_cnt_large_cut, cv2.COLOR_BGR2RGB))

#%% test wether includes fine features and do segmentation
X = img[mask == 255]
def find_segments(X, area_thresh = 400):
    variance = np.std(X, axis = 0)
    # kmeans = KMeans(n_clusters = 2, n_init = 3)
    segments = []
    seg_further_list = []
    if (variance < 5).all():
        segments.append(X)
    else:
        seg_done, seg_further = fine_segment(X, area_thresh)
        segments += seg_done
        seg_further_list += seg_further
        index = 0
        while index < len(seg_further_list):
            seg_done, seg_further = fine_segment(seg_further_list[index], area_thresh)
            segments += seg_done
            seg_further_list += seg_further
            index += 1
    print(len(segments), [np.mean(segments[i], axis = 0) for i in range(len(segments))])
    return segments

def careful_look(data, segments, area_thresh = 400):
    real_flake_list = []
    for seg in segments:
        mean = np.mean(seg, axis = 0)
        
        ret, thickness, contrast = calculate_contrast(mean, local_bk_color)
        print('thickness', thickness, 'contrast', contrast)
        if not ret:
            print('out of range')
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
            if area > area_thresh:
                print('area: ', area)
                mask = np.zeros(img_open.shape[:2], np.uint8)
                cv2.drawContours(mask, [contours[i]], -1, 255, -1)
                piece = data[mask==255]
                variance = np.std(piece, axis = 0)
                print(variance)
                # if not (variance < 5).all():
                #     continue
                mean = np.mean(piece, axis = 0)
                ret, thickness, contrast = calculate_contrast(mean, local_bk_color)
                
                
                if ret:
                    real_flake_list.append(contrast)

                plt.figure()
                piece = cv2.bitwise_and(data, data, mask = mask)
                plt.imshow(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))
                break
        # print(contrast)
        # plt.figure()
        # plt.imshow(img_open, cmap = 'gray')
        

    if len(real_flake_list) > 0:
        return True, real_flake_list
    return False, []

segments = find_segments(X)
careful_look(img_cnt_large_cut, segments)
#%%
def fine_segment(data, area_thresh = 400):
    seg_done = []
    seg_further = []
    # labels = kmeans.fit_predict(data)
    kmeans = faiss.Kmeans(d=3, k=2)
    kmeans.train(data.astype(np.float32))
    labels = kmeans.index.search(data.astype(np.float32), 1)[1].squeeze()
    for i in range(2):
        seg = data[labels == i]
        if len(seg) < area_thresh:
            continue
        variance = np.std(seg, axis = 0)
        if (variance < 5).all():
            seg_done.append(seg)
        else:
            seg_further.append(seg)
    return seg_done, seg_further


def calculate_contrast(sample_color, bk_color):
    '''
    color: np.array in the order of blue, green, red (BGR)
    '''
    contrast_range = np.array([[-0.25, 0.15], [0.04, 0.8], [0.04, 0.8]])
    contrast = (bk_color - sample_color) / bk_color
    ret, thickness, _ = predict_gr(contrast)
    # for i in range(3):
    #     if not contrast_range[i, 0] <= contrast[i] <= contrast_range[i, 1]:
    #         return False, contrast
    return ret, thickness, contrast


def get_local_bk(data):
    bk_color_local = np.zeros(3)
    hist = [0, 0, 0]
    delta = 5
    for i in range(3):
        hist[i] = cv2.calcHist([data[:, :, i]], [0], None, [256], [0,255])
        hist[i] = hist[i][bk_color[i] - delta: bk_color[i] + delta + 1]
        bk_color_local[i] = np.argmax(hist[i]) + bk_color[i] - delta
    return bk_color_local



#%%
img_cnt_large_cut = img_cnt_large_cut.astype(np.float64)
for i in range(3):
    img_cnt_large_cut[:, :, i] -= np.min(img_cnt_large_cut[:, :, i]) / 2
    img_cnt_large_cut[:, :, i] *= 255/np.max(img_cnt_large_cut[:, :, i]) - 0.01
img_cnt_large_cut = img_cnt_large_cut.astype(np.uint8)

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint="D:/Python/sam/sam_vit_b_01ec64.pth")
sam.to(device = device)


#%%
start = time.time()
# mask_generator = SamAutomaticMaskGenerator(sam, points_per_side = 4,
# stability_score_thresh = 0.8, pred_iou_thresh = 0.8)
mask_generator = SamAutomaticMaskGenerator(sam, )
start = time.time()
masks = mask_generator.generate(img)
print(time.time() - start, len(masks))


# %%
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns[0:]:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.7]])
        img[m] = color_mask
    ax.imshow(img)
# %%
quantized_image = cv2.pyrMeanShiftFiltering(img_cnt_large_cut, 20, 40)
gray_image = cv2.cvtColor(img_cnt_large_cut, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image, 20, 50)
kernel = np.ones((2, 2), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)
contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros(img_cnt_large_cut.shape[:2], np.uint8)
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
# segmented_image = cv2.bitwise_and(image, image, mask=mask)
plt.figure()
plt.imshow(mask)
# %%

kmeans = faiss.Kmeans(d=x_train.shape[1], k=k, niter=max_iter, nredo=n_init)
kmeans.train(x_train.astype(np.float32))

#%%
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


img_binary = cv2.inRange(img_gray, 0, 70)

kernel_open = np.ones((10,10),np.uint8)
img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel_open)

kernel_dilation = np.ones((20, 20), np.uint8)
img_dilation = cv2.dilate(img_close, kernel_dilation, iterations=1)

img[img_dilation==255] = 0
# plt.figure()
# plt.plot(hist_y)
plt.figure()
# plt.imshow(img_gray)
# plt.imshow(img_binary, cmap = 'gray')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# %%
