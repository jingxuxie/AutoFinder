
#%%
import cv2
import os
import json
from shutil import copy2
from autofinder.auxiliary_func import get_background, generate_positions, \
merge_thumbnail, generate_revisit_list, draw_scale, pre_process, locate_in_large
from autofinder.find_layers.find_gr import layer_search_gr
from autofinder.find_layers.find_bn import layer_search_bn
from autofinder.find_layers.find_thin_bn import layer_search_thin_bn
from autofinder.find_layers.find_ws2 import layer_search_ws2
from autofinder.find_layers.find_wse2 import layer_search_wse2
from autofinder.find_layers.find_mose2 import layer_search_mose2
from autofinder.combine_RGB import combine_rgb
import numpy as np



#%%
def start_search(raw_folder, color_folder, output_folder, contour_index, material = 'Graphene', 
                 area = 1000, combine_reso = 'normal', substrate = '90nm', 
                 thickness_range = [0, 10], revisit_magnification = '50x'):
    
    combine_rgb(raw_folder, color_folder)
    input_folder = color_folder
    scale, bk = pre_process(input_folder, combine_reso)

    _, file_list, _ = generate_positions(input_folder)
    contrast_json = {'items': []}
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    for i in range(len(file_list)):
        filename = file_list[i]
        print(i, filename)

        input_name = input_folder + '/' + filename
        result_name = output_folder + '/' + filename[:-3] + 'jpg'

        search_function = get_search_funtion(material)
        ret, img_out, contrast_list, flake_position_list = search_function(input_name, bk, area, thickness_range)
        # print(area)
        
        if ret:
            print(ret)
            img_out = draw_scale(img_out, 10, img_out.shape[1], img_out.shape[0], calibration = 0.43945)
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

    revisit_pos, file_conterpart = generate_revisit_list(input_folder, output_folder, 
                                                         revisit_magnification = revisit_magnification)
    np.save(output_folder + '/revisit_pos.npy', revisit_pos)
    with open(output_folder + '/file_conterpart.json', 'w') as f:
        f.write(json.dumps(file_conterpart))
    
    params = {'input_folder':input_folder, 'output_folder':output_folder, 'contour_index':contour_index,
              'material':material, 'area': area, 'combine_reso':combine_reso, 'substrate':substrate, 
              'thickness_range':thickness_range}
    with open(output_folder + '/params.json', 'w') as f:
        f.write(json.dumps(params))





def get_search_funtion(material = 'Graphene'):
    if material == 'Graphene':
        func = layer_search_gr
    elif material == 'hBN':
        func = layer_search_bn
    elif material == 'thin hBN':
        func = layer_search_thin_bn
    elif material =='WS2':
        func = layer_search_ws2
    elif material == 'WSe2':
        func = layer_search_wse2
    elif material == 'MoSe2':
        func = layer_search_mose2
    else:
        raise NotImplementedError
    return func






#%%


