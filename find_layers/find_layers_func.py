#%%
import numpy as np
import faiss
import cv2
import scipy

#%%
def calculate_contrast(sample_color, bk_color, predictor):
    '''
    color: np.array in the order of blue, green, red (BGR)
    '''
    contrast = (bk_color - sample_color) / bk_color
    ret, thickness, _ = predictor(contrast, bk_color)
    return ret, thickness, list(contrast)




def get_local_bk(data, bk_color):
    bk_color_local = np.zeros(3)
    hist = [0, 0, 0]
    delta = 5
    for i in range(3):
        hist[i] = cv2.calcHist([data[:, :, i]], [0], None, [256], [0,255])
        hist[i] = hist[i][bk_color[i] - delta: bk_color[i] + delta + 1]
        bk_color_local[i] = np.argmax(hist[i]) + bk_color[i] - delta
    return bk_color_local




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
        mean = np.mean(seg, axis = 0)
        variance_limit = mean * 0.05
        variance_limit = np.maximum(np.zeros(3) + 2, variance_limit)
        variance_limit = np.maximum(np.zeros(3) + 7, variance_limit)
        if (variance < variance_limit).all():
            seg_done.append(seg)
        else:
            seg_further.append(seg)
    return seg_done, seg_further




def find_segment_list(X, area_thresh = 200, variance_limit = np.array([6, 6, 6])):
    variance = np.std(X, axis = 0)
    mean = np.mean(X, axis = 0)
    variance_limit = mean * 0.05
    variance_limit = np.maximum(np.zeros(3) + 2, variance_limit)
    variance_limit = np.maximum(np.zeros(3) + 7, variance_limit)
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




def careful_look(data, segments, local_bk_color, predictor, area_thresh = 400, 
                 thickness_range_list = [[0, 50]]):
    real_flake_list = []
    thickness_list = []
    for seg in segments:
        mean = np.mean(seg, axis = 0)
        ret, thickness, contrast = calculate_contrast(mean, local_bk_color, predictor)
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
                # variance = np.std(piece, axis = 0)
                # if not (variance < 5).all():
                #     continue
                mean = np.mean(piece, axis = 0)
                ret, thickness, contrast = calculate_contrast(mean, local_bk_color, predictor)
                count = 0
                for thickness_range in thickness_range_list:
                    if not thickness_range[0] <= thickness <= thickness_range[1]:
                        count += 1
                if count == len(thickness_range_list):
                    continue
                if ret:
                    real_flake_list.append(contrast)
                    thickness_list.append(thickness)
                    break
        # print(contrast)
        # plt.figure()
        # plt.imshow(img_open, cmap = 'gray')
        # print(ret, thickness, thickness_list)
        

    if len(real_flake_list) > 0:
        return True, real_flake_list, thickness_list
    return False, [], []




def put_Text(img, thickness_list, boundingRec):
    x, y, w, h = boundingRec
    for i in range(len(thickness_list)):
        thickness = round(thickness_list[i], 1)
        middle = img.shape[1] / 2
        if abs(x - w -  240 - middle) < abs(x + 2*w - middle):
            pos_x = x - w - 240
        else:
            pos_x = x + 2*w
        pos_y = max(100, y - h + 50)
        cv2.putText(img, str(thickness) + 'nm', (pos_x, pos_y + 60*i), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 255), 2, cv2.LINE_AA)


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
# %%
