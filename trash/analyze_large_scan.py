#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

#%%
def combine(folder, crop_w = [0, 1], crop_h = [0, 1], scale = 0.2, compress = 1):
    pos_file = folder + '/position.txt'

    with open(pos_file) as f:
        lines = f.readlines()

    file_list = []
    position_list = []
    for line in lines:
        char = line.split()
        file_list.append(char[0])
        pos = np.array([float(char[1]), float(char[2])])
        position_list.append(pos)

    position_list = np.array(position_list)

    x_limit = [np.min(position_list[:, 0]), np.max(position_list[:, 0])]
    y_limit = [np.min(position_list[:, 1]), np.max(position_list[:, 1])]

    position_list[:, 0] -= x_limit[0]
    position_list[:, 1] -= y_limit[0]

    # position_list, file_list = normalize_position(position_list, file_list)

    x_range = x_limit[1] - x_limit[0]
    y_range = y_limit[1] - y_limit[0]

    
    # bk = cv2.imread('C:/Jingxu/10x.png')
    # bk = np.array(bk, dtype = float)
    # median = np.zeros(3) + 1
    # for i in range(3):
    #     median[i] = np.median(bk[:, :, i])


    img = cv2.imread(folder + '/' + file_list[0])

    width = int(img.shape[1] / compress)
    height = int(img.shape[0] / compress)

    out = np.zeros((int(y_range * scale + height + 100), 
                    int(x_range * scale + width + 100), 
                    3), dtype = np.uint8)

    roi_w = [int(width * crop_w[0]), int(width * crop_w[1])]
    roi_h = [int(height * crop_h[0]), int(height * crop_h[1])]

    roi_width = roi_w[1] - roi_w[0]
    roi_height = roi_h[1] - roi_h[0]

    for i in range(len(file_list)):
        img = cv2.imread(folder + '/' + file_list[i])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height))
        pos = position_list[i]
        start_x = int(pos[0] *scale)
        start_y = int(pos[1] *scale) + 1
        out[-start_y - roi_height: -start_y, start_x: start_x + roi_width] = img[roi_h[0]: roi_h[1], roi_w[0]: roi_w[1]]
    cv2.imwrite(folder + '/combine.jpg', out)
    return out
    


#%%
def find_contours(img, bright_thresh = 50, area_thresh = 500):
    img = cv2.resize(img, (int(img.shape[1]/10), int(img.shape[0]/10)))
    img = np.flip(img, axis = 0)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, bright_thresh, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_list = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_thresh:
            cnt_list.append(cnt)
    # cv2.drawContours(img, cnt_list, -1, (0,255,0), 3)

    # plt.figure()
    # plt.imshow(img)
    return cnt_list



#%%
def generate_points_for_chips(polygon, grid_spacing, margin):
    buffered_polygon = polygon.buffer(margin)
    minx, miny, maxx, maxy = buffered_polygon.bounds  # Bounding box of the polygon
    points = []

    snake_direction = 1  # Direction indicator for the snake-style pattern
    for x in np.arange(minx, maxx, grid_spacing):
        if snake_direction == 1:
            y_range = np.arange(miny, maxy, grid_spacing)
        else:
            y_range = np.arange(maxy, miny, -grid_spacing)
        
        for y in y_range:
            point = Point(x, y)
            if buffered_polygon.contains(point):  # Check if the point is inside the buffered polygon
                points.append([x, y])

        snake_direction *= -1  # Reverse the direction for the next row

    return np.array(points)

#%%
def generate_small_scan_position(cnt_list, grid_spacing, margin):
    grid_points_list= []
    for cnt in cnt_list:
        cnt = np.squeeze(cnt, axis = 1)

        polygon_vertices = cnt
        polygon = Polygon(polygon_vertices)

        grid_points = generate_points_for_chips(polygon, grid_spacing, margin)
        grid_points_list.append(grid_points)

    return grid_points_list

    
    # plt.scatter(grid_points[:, 0], grid_points[:, 1])
    # plt.scatter(cnt[:, 0], cnt[:, 1])
#%%
def analyze_large_scan(folder, compress = 1, scale = 0.125, grid_spacing = 12.5, margin = 8):
    img_large = combine(folder, compress = compress, scale = scale)
    cnt_list = find_contours(img_large)
    grid_points_list = generate_small_scan_position(cnt_list, grid_spacing, margin)
    for i in range(len(grid_points_list)):
        grid_points_list[i] /= (scale/10)
        grid_points_list[i][:, 0] -= 2000
        grid_points_list[i][:, 1] -= 1000
    return grid_points_list





#%%
def normalize_position(position_list, file_list):
    out_position = [position_list[0]]
    out_file = [file_list[0]]
    i = 1
    while i < len(position_list):
        if position_list[i][1] > position_list[i - 1][1]:
            out_position.append(position_list[i])
            out_file.append(file_list[i])
            i += 1
        else:
            temp_pos = [position_list[i]]
            temp_file = [file_list[i]]
            j = 1
            while i + j < len(position_list):
                if position_list[i + j][1] < position_list[i + j - 1][1]:
                    temp_pos.append(position_list[i + j])
                    temp_file.append(file_list[i + j])
                    j += 1
                else:
                    break
            # print(temp_pos)
            # print(temp_pos.reverse())
            temp_pos.reverse()
            temp_file.reverse()

            out_position += temp_pos
            out_file += temp_file

            i += j

    return out_position, out_file

            





# %%
if __name__ == '__main__':
    # out = combine('F:/Temp/raw', scale = 0.125)
    # plt.figure()
    # plt.imshow(out)

    points_list = analyze_large_scan('F:/Temp/raw')
    plt.figure()
    for points in points_list:
        plt.scatter(points[:, 0], points[:, 1])
# %%
