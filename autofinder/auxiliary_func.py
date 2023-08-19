# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:36:44 2020

@author: HP
"""
#%%
import numpy as np
from numba import jit
import os
from autofinder.BresenhamAlgorithm import Pos_of_Line
from PyQt5.QtGui import QImage
import json
import cv2
from shutil import copyfile, copy2

#%%
@jit(nopython = True)
def go_fast(a,b,c):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            temp = int(a[i,j]+b)*c
            temp = max(0, temp)
            temp = min(temp,255)
            a[i,j] = temp
    return a


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
def matrix_divide(a,b):
    if len(a.shape) == 3:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                for k in range(a.shape[2]):
                    a[i,j,k] = round(a[i,j,k]/(b+0.001))
    return a



def get_folder_from_file(filename):
    folder = filename
    while folder[-1] != '/':
        folder = folder[:-1]
    return folder

def get_filename_from_path(pathname):
    path = pathname
    i = -1
    while path[i] != '/':
        i -= 1
    return path[i+1:]

def get_position_from_string(string):
    previous_index = 0
    count = 0
    x, y, z = 0, 0, 0
    for i in range(len(string)):
        if string[i] == ' ':
            if count == 0:
                try:
                    x = int(string[:i])
                except:
                    return False, [0, 0, 0]
                else:
                    previous_index = i
                    count += 1
            elif count == 1:
                try:
                    y = int(string[previous_index:i])
                    z = int(string[i:])
                except:
                    return False, [0,0,0]
                else:
                    return True, [x, y, z]
                break
    
            
    

@jit(nopython = True)
def float2uint8(img_aver):

    if len(img_aver.shape) == 3:
        for i in range(img_aver.shape[0]):
            for j in range(img_aver.shape[1]):
                for k in range(img_aver.shape[2]):
                    #img_aver[i,j,k] = round(img_aver[i,j,k])
                    img_aver[i,j,k] = max(0, img_aver[i,j,k])
                    img_aver[i,j,k] = min(255, img_aver[i,j,k])
    
    '''
    elif len(img_aver.shape) == 1:
        for i in range(img_aver.shape[0]):
            for j in range(img_aver.shape[1]):
                img_aver[i,j] = max(0, img_aver[i,j])
                img_aver[i,j] = min(255, img_aver[i,j])
    '''
    return img_aver.astype(np.uint8)



#@jit(nopython = True)
def calculate_contrast(matrix, x1_1, y1_1, x1_2, y1_2, x2_1, y2_1, x2_2, y2_2):
    group1_x, group1_y = Pos_of_Line(x1_1, y1_1, x1_2, y1_2)
    group2_x, group2_y = Pos_of_Line(x2_1, y2_1, x2_2, y2_2)
    color1 = np.zeros(4)
    color2 = np.zeros(4)
    #if len(matrix.shape) == 3:
        #matrix = matrix[:,:,1]
    if len(group1_x)>5:
        group1_x = group1_x[:-3]
        group1_y = group1_y[:-3]
        for i in range(len(group1_x)):
            x = min(group1_y[i], matrix.shape[0] - 1)
            y = min(group1_x[i], matrix.shape[1] - 1)
            if len(matrix.shape) == 3: 
                color1[:-1] += matrix[x, y] #*0.299 + matrix[x, y, 1]*0.587 + matrix[x, y, 2]*0.114)
            else:
                color1[-1] += matrix[x, y]
        color1 /= len(group1_x)
    else:
        return 0, np.zeros(3)
    
    if len(group2_x)>5:
        group2_x = group2_x[:-3]
        group2_y = group2_y[:-3]
        for j in range(len(group2_x)):
            x = min(group2_y[j], matrix.shape[0] - 1)
            y = min(group2_x[j], matrix.shape[1] - 1)
            if len(matrix.shape) == 3: 
                color2[:-1] += matrix[x, y] #*0.299 + matrix[x, y, 1]*0.587 + matrix[x, y, 2]*0.114)
            else:
                color2[-1] += matrix[x,y]
        color2 /= len(group2_x)
    else:
        return 0, np.zeros(3)
    
    #print (color1, color2)
    
    contrast = (color1 - color2) / (color1 + 0.001)
    if len(matrix.shape) == 3:
        contrast[-1] = contrast[0] * 0.299 + contrast[1] * 0.587 + contrast[2] * 0.114
    return contrast[-1], contrast[:-1]



@jit(nopython = True)
def record_draw_shape(blank, x_pos, y_pos, num):
    if len(x_pos) > 0:
        for i in range(len(x_pos)):
            if 0 <= x_pos[i] < blank.shape[1] and 0 <= y_pos[i] < blank.shape[0]:
                
                blank[y_pos[i], x_pos[i]] = num
            
    return blank
    


def calculate_angle(pos11,pos12,pos21,pos22):
    x11, y11 = list(pos11)[0], list(pos11)[1]
    x12, y12 = list(pos12)[0], list(pos12)[1]
    x21, y21 = list(pos21)[0], list(pos21)[1]
    x22, y22 = list(pos22)[0], list(pos22)[1]
    
    a_square = (x12 - x11)**2 + (y12 - y11)**2
    b_square = (x22 - x21)**2 + (y22 - y21)**2
    a = np.sqrt(a_square)
    b = np.sqrt(b_square)
    c_square = ((x12 - x11)-(x22 - x21))**2 + ((y12 - y11)-(y22 - y21))**2
    if a*b == 0 or a == b:
        return 0
    else:
        theta = (a_square + b_square - c_square)/(2 * a * b)
        if abs(theta) > 1:
            return 0
        theta = np.arccos(theta)
        theta = theta/np.pi * 180
    return theta

def np2qimage(img):
    if len(img.shape)==3:
        qimage = QImage(img[:], img.shape[1], img.shape[0],\
                      img.shape[1] * 3, QImage.Format_RGB888)
    else:
        qimage = QImage(img[:], img.shape[1], img.shape[0],\
                      img.shape[1] * 1, QImage.Format_Indexed8)
    return qimage

#%%
def generate_grid_points(x_step = 2000, y_step = 2000, x_num = 10, y_num = 10):
        position = [[0, 0, 0]]
        for i in range(x_num + 1):
            if i % 2 == 0:
                position += [[0, y_step, 0] for j in range(y_num)]
            else:
                position += [[0, -y_step, 0] for j in range(y_num)]
            position.append([x_step, 0, 0])

        print(position)
        return position[:-1]

#%%
def generate_revisit_list(input_folder, output_folder, x_margin = 1728, y_margin = 1000,
                          x_shift = 576, y_shift = 0, scale = 0.4954, img_height = 2160,
                          FOV = 200):
    file_pos_dict, _, _ = generate_positions(input_folder)
    with open(output_folder + '/flakes_info.json') as json_file:
        data = json.load(json_file)
    items = data['items']
    positions = []
    file_conterpart_list = {'files':[]}

    for item in items:
        filename = item['filename']
        parent_pos = file_pos_dict[filename]
        for pos in item['flake_position_list']:
            correction = np.array([x_shift - x_margin, y_shift - y_margin])
            relative_pos = scale * (np.array([pos[0] + pos[2]/2, img_height - (pos[1] + pos[3]/2)]) + correction)
            pos = parent_pos + relative_pos
            ret = check_within_FOV(positions, pos, radius = FOV)
            if not ret:
                positions.append(pos)
                file_conterpart_list['files'].append(filename)
    return np.array(positions), file_conterpart_list



#%%
def check_within_FOV(existing_list, new_pos, radius):
    if len(existing_list) == 0:
        return False
    existing_list = np.array(existing_list)
    diff = existing_list - new_pos
    distance = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
    if np.min(distance) < radius:
        return True
    return False



#%%
def merge_thumbnail(folder, filename, img, scale = 0.25):
    file_pos_dict, _, position_list = generate_positions(folder)
    thumbnail = cv2.imread(folder + '/combine.jpg')
    x_limit = np.min(position_list[:, 0])
    y_limit = np.min(position_list[:, 1])
    pos = file_pos_dict[filename] - np.array([x_limit, y_limit])
    pos += np.array([500, 500])

    ratio = min(1500 / thumbnail.shape[0], 1500 / thumbnail.shape[1])
    thumbnail_size = np.int32(np.array(thumbnail.shape[:2]) * ratio)
    thumbnail = cv2.resize(thumbnail, np.flip(thumbnail_size))
    
    pos = np.int32([pos[0] *scale * ratio, thumbnail_size[0] - pos[1] * scale *ratio])
    
    cv2.circle(thumbnail, pos, int(1000*scale*ratio), (0, 0, 255), 3)
    out = np.zeros((img.shape[0], img.shape[1] + thumbnail_size[1], 3), dtype = np.uint8)
    out[:img.shape[0], :img.shape[1]] = img
    out[:thumbnail.shape[0], img.shape[1]:] = thumbnail
    # out[:, img.shape[1]: img.shape[1] + 20, :] = 255
    return out



#%%
def get_background(folder, width = 1920, height = 1080):
    _, file_list, _ = generate_positions(folder)
    file_num = len(file_list)
    matrix = np.zeros((file_num, height, width, 3), dtype = np.uint8)
    for i in range(file_num):
        filename = file_list[i]
        input_name = folder + '/' + filename
        img = cv2.imread(input_name)
        img = cv2.resize(img, (width, height))
        matrix[i] = img
    background = np.median(matrix, axis = 0)
    # median = np.median(background, axis = 0)
    return background

#%%
def generate_positions(folder):
    pos_file = folder + '/position.txt'

    with open(pos_file) as f:
        lines = f.readlines()

    file_pos_dict = {}
    file_list = []
    position_list = []
    for line in lines:
        char = line.split()
        filename = char[0]
        
        pos = np.array([float(char[1]), float(char[2])])
        file_pos_dict[filename] = pos
        file_list.append(filename)
        position_list.append(pos)

    position_list = np.array(position_list)
    
    return file_pos_dict, file_list, position_list


#%%
def crop_quarter_of_20(origin_folder, input_folder, output_folder = None,
                       crop_w = [0.2, 0.7], crop_h = [0.3, 0.8]):
    _, file_list, _ = generate_positions(input_folder)
    if output_folder != None:
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

    for file in file_list:
        filename = input_folder + '/' + file
        img = cv2.imread(filename)

        width = img.shape[1]
        height = img.shape[0]

        roi_w = [int(width * crop_w[0]), int(width * crop_w[1])]
        roi_h = [int(height * crop_h[0]), int(height * crop_h[1])]

        img = img[roi_h[0]: roi_h[1], roi_w[0]: roi_w[1]]
        img = draw_scale(img, 40, img.shape[1], img.shape[0], calibration = 0.9910185)
        img = merge_thumbnail_top_right(origin_folder, input_folder, file, img)

        if output_folder == None:
            cv2.imwrite(filename, img)
        else:
            cv2.imwrite(output_folder + '/' + file, img)
        copy2(input_folder + '/position.txt', output_folder)


def merge_thumbnail_top_right(origin_folder, folder, filename, img, scale = 0.5):
    _, _, origin_position_list = generate_positions(origin_folder)
    x_limit = np.min(origin_position_list[:, 0])
    y_limit = np.min(origin_position_list[:, 1])

    file_pos_dict, _, position_list = generate_positions(folder)
    thumbnail = cv2.imread(folder + '/combine.jpg')
    
    pos = file_pos_dict[filename] - np.array([x_limit, y_limit])
    pos += np.array([500, 500])

    ratio = min(500 / thumbnail.shape[0], 500 / thumbnail.shape[1])
    thumbnail_size = np.int32(np.array(thumbnail.shape[:2]) * ratio)
    thumbnail = cv2.resize(thumbnail, np.flip(thumbnail_size))
    
    pos = np.int32([pos[0] *scale * ratio, thumbnail_size[0] - pos[1] * scale *ratio])
    
    cv2.circle(thumbnail, pos, int(1000*scale*ratio), (0, 0, 255), 3)
    out = img
    out[:thumbnail.shape[0], -thumbnail.shape[1]:] = thumbnail
    return out


def add_signature(folder):
    files = os.listdir(folder)
    for file in files:
        if not file[-3:] in ['jpg', 'png']:
            continue
        filename = folder + '/' + file
        img = cv2.imread(filename)
        fontscale = img.shape[0]/1000
        thickness = round(fontscale)
        x_shift = int(275 * fontscale)
        y_shift = int(20 * fontscale)
        cv2.putText(img, 'Designed by Jingxu', (img.shape[1] - x_shift, img.shape[0] - y_shift), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 
                color = (150, 150, 150), fontScale = fontscale, thickness = thickness)
        cv2.imwrite(filename, img)

#%%
def merge_folder(input_folder, output_folder):
    _, old_file_list, _ = generate_positions(input_folder)
    with open(output_folder + '/file_conterpart.json') as json_file:
        file_conterpart = json.load(json_file)['files']
    num = len(old_file_list)
    assert num == len(file_conterpart)

    count = 1
    for i in range(num):
        if i > 1:
            if file_conterpart[i] == file_conterpart[i - 1]:
                count += 1
            else:
                count = 1
        old_filename = input_folder + '/' + old_file_list[i]
        new_filename = output_folder + '/' + file_conterpart[i][:-(4+len(str(count)))] + \
                       str(count) + file_conterpart[i][-4:]
        copyfile(old_filename, new_filename)

def copy_image(input_folder, output_folder):
    files = os.listdir(input_folder)
    for file in files:
        if file[-3:] in ['jpg', 'png']:
            filename = input_folder + '/' + file
            copy2(filename, output_folder)

#%%
def draw_scale(img, magnification, width, height, calibration = 0.9910185):
    '''
    calibration: um / pixel under 10x objective
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(1, img.shape[0] / 1000)
    ratio = calibration / magnification * 10
    # width = ratio
    # height *= ratio
    # print(width)
    dimension = max(width*ratio, height*ratio)
    if dimension > 2000:
        unit = 400
    elif dimension > 1000:
        unit = 200
    elif dimension > 500:
        unit = 100
    elif dimension > 200:
        unit = 40
    elif dimension >100:
        unit = 20
    elif dimension > 50:
        unit = 10
    else:
        unit = 5
    unit_for_img = round(unit/ratio)
    cv2.line(img, (0, int(2*scale)), (width, int(2*scale)), (0, 0, 255), int(3*scale))
    cv2.line(img, (int(2*scale), 0), (int(2*scale), height), (0, 0, 255), int(3*scale))
    
    scale_for_img = unit_for_img
    scale_for_text = unit
    cv2.putText(img, str(magnification)+'x', (int(5*scale), int(25*scale)), font, 0.7*scale, (0,0,255), int(2*scale), cv2.LINE_AA)
    while scale_for_img < width:
        cv2.line(img, (scale_for_img, 0), (scale_for_img, int(15*scale)), (0,0,255), int(2*scale))
        text = str(scale_for_text)
        cv2.putText(img, text, (scale_for_img-int(15*scale), int(30*scale)), font, 0.5*scale, (0,0,255), int(1*scale), cv2.LINE_AA)
        scale_for_img += unit_for_img
        scale_for_text += unit
    
    scale_for_img = unit_for_img
    scale_for_text = unit
    while scale_for_img < height:
        cv2.line(img, (0, scale_for_img), (int(15*scale), scale_for_img), (0,0,255), int(2*scale))
        text = str(scale_for_text)
        cv2.putText(img, text, (int(5*scale), scale_for_img-int(5*scale)), font, 0.5*scale, (0,0,255), int(1*scale), cv2.LINE_AA)
        scale_for_img += unit_for_img
        scale_for_text += unit

    return img

#%%
if __name__ == '__main__':
    u = os.path.abspath(__file__).replace('\\','/')
    u = get_folder_from_file(u)