#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from autofinder.auxiliary_func import background_divide, generate_positions, combine


#%%
def find_contours(img, bright_thresh = 10, area_thresh = 1000):
    img = cv2.resize(img, (int(img.shape[1]/10), int(img.shape[0]/10)))
    # img = np.flip(img, axis = 0)
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
    minx = max(0, minx)
    miny = max(0, miny)
    points = []

    snake_direction = 1  # Direction indicator for the snake-style pattern
    for y in np.arange(miny, maxy, grid_spacing):
        if snake_direction == 1:
            x_range = np.arange(minx, maxx, grid_spacing)
        else:
            x_range = np.arange(maxx, minx, -grid_spacing)
        
        for x in x_range:
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
def analyze_large_scan(folder, compress = 1, scale = 0.149, grid_spacing = 17.91, margin = 13):
    img_large = combine(folder, compress = compress, scale_x = scale, scale_y = scale)
    cnt_list = find_contours(img_large)
    grid_points_list = generate_small_scan_position(cnt_list, grid_spacing, margin)
    for i in range(len(grid_points_list)):
        grid_points_list[i] /= (scale/10)
        grid_points_list[i][:, 0] -= 1200
        grid_points_list[i][:, 1] -= 1200
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