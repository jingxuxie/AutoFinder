
#%%
import cv2
import os
import json
from shutil import copy2
from autofinder.auxiliary_func import get_background, generate_positions, \
merge_thumbnail, generate_revisit_list, draw_scale
from autofinder.analyze_large_scan import combine
from autofinder.find_layers.find_gr import layer_search_gr
from autofinder.find_layers.find_bn import layer_search_bn
from autofinder.find_layers.find_ws2 import layer_search_ws2
from autofinder.find_layers.find_wse2 import layer_search_wse2
import numpy as np



#%%
def start_search(input_folder, output_folder, contour_index, material = 'Graphene', 
                 area = 1000, combine_reso = 'normal', substrate = '90nm', 
                 thickness_range = [0, 50]):
    bk = get_background(input_folder, width = 1920, height = 1080)
    bk = cv2.resize(bk, (3840, 2160))
    
    scale, compress = get_scale_compress(combine_reso)
    combine(input_folder, crop_w = [0.15, 0.75], scale = scale, compress = compress, bk = bk)

    _, file_list, _ = generate_positions(input_folder)
    contrast_json = {'items': []}
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    for i in range(len(file_list)):
        filename = file_list[i]
        print(i, filename)

        input_name = input_folder+'/'+filename
        result_name = output_folder+'/'+filename

        search_function = get_search_funtion(material)
        ret, img_out, contrast_list, flake_position_list = search_function(input_name, bk, area, thickness_range)
        print(area)
        
        if ret:
            print(ret)
            img_out = draw_scale(img_out, 10, img_out.shape[1], img_out.shape[0], calibration = 0.49551)
            out = merge_thumbnail(input_folder, filename, img_out, scale = scale)
            cv2.imwrite(result_name, out)
            item = {'filename': filename,
                    'contrast_list': contrast_list,
                    'flake_position_list': flake_position_list}
            contrast_json['items'].append(item)
            
    a = json.dumps(contrast_json)
    with open(output_folder + '/flakes_info.json', 'w') as f:
        f.write(a)

    copy2(input_folder + '/combine.jpg', output_folder)
    locate_in_large(input_folder, output_folder, contour_index)

    revisit_pos, file_conterpart = generate_revisit_list(input_folder, output_folder)
    np.save(output_folder + '/revisit_pos.npy', revisit_pos)
    with open(output_folder + '/file_conterpart.json', 'w') as f:
        f.write(json.dumps(file_conterpart))
    
    params = {'input_folder':input_folder, 'output_folder':output_folder, 'contour_index':contour_index,
              'material':material, 'area': area, 'combine_reso':combine_reso, 'substrate':substrate, 
              'thickness_range':thickness_range}
    with open(output_folder + '/params.json', 'w') as f:
        f.write(json.dumps(params))

    
def get_scale_compress(combine_reso = 'normal'):
    if combine_reso == 'low':
        scale = 0.25
        compress = 8
    elif combine_reso == 'normal':
        scale = 0.5
        compress = 4
    elif combine_reso == 'hi':
        scale = 1
        compress = 2
    return scale, compress



def get_search_funtion(material = 'Graphene'):
    if material == 'Graphene':
        func = layer_search_gr
    elif material == 'hBN':
        func = layer_search_bn
    elif material =='WS2':
        func = layer_search_ws2
    elif material == 'WSe2':
        func = layer_search_wse2
    return func




def locate_in_large(input_folder, output_folder, contour_index):
    img_large = cv2.imread(os.path.dirname(input_folder) + '/large/combine.jpg')
    img_large = np.flip(img_large, axis = 0)
    cnt = find_contours(img_large, contour_index)
    x, y, w, h = cv2.boundingRect(cnt)
    img_large = np.float32(img_large)
    cv2.rectangle(img_large, (x, y), (x+w, y+h), (0, 0, 255), 30)
    cv2.imwrite(output_folder + '/large.jpg', np.flip(img_large, axis = 0))



#%%
def find_contours(img, contour_index, bright_thresh = 50, area_thresh = 1e5):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, bright_thresh, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_list = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_thresh:
            cnt_list.append(cnt)
            # print(area)
    return cnt_list[contour_index]

# %%
