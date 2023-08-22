#%%
import numpy as np
import time
from autofinder.scan import Scanner
from autofinder.analyze_large_scan import analyze_large_scan, combine
from autofinder.stages.stage_rpi import Stage_Rpi
from autofinder.stages.stage_mcx_e1000 import Stage_MCX_E1000
from autofinder.lights.RGB_LED import RGBW_LED
import time
import cv2
from autofinder.Camera import Camera
from autofinder.cameras.vieworks import VieworksCamera
import matplotlib.pyplot as plt
import threading
import os
from shutil import copy2
from autofinder.search_layers import start_search
from autofinder.auxiliary_func import crop_quarter_of_20, add_signature, \
merge_folder, copy_image, generate_grid_points, analyze_revisit, process_scan_only
#%%
class Scan_Platform():
    def __init__(self, camera, stage, light):
        self.scanner = Scanner(camera, stage, light)
        self.stage = self.scanner.stage
        self.camera = self.scanner.camera
        self.autofocus = self.scanner.autofocus

    def initialize(self):
        ret = self.scanner.initialize()
        if ret:
            return True
        return False
    
    def start_scan_large(self, x_step = 2400, y_step = 2400, x_num = 31, y_num = 21):
        self.camera.set_exposure_time(1000)
        self.stage.goto_xyz(20000, 0)
        self.stage.move_turret(1)
        time.sleep(2)
        self.stage.home()

        position = generate_grid_points(x_step, y_step, x_num, y_num)
        self.scanner.scan(position, subfolder = 'large', compress = 8, delay = 0.1, RGB_color = False)
        self.grid_points_list = analyze_large_scan(self.scanner.save_folder)

    def start_scan_small(self, fast = False, scan_only = False, **kwargs):
        self.stage.move_turret(2)
        time.sleep(1)
        num = len(self.grid_points_list)
        self.search_thread = [[] for i in range(num)]
        for i in range(num):
            points = self.grid_points_list[i]
            self.stage.goto_xyz(*points[0])
            time.sleep(0.5)
            self.scanner.scan(points, focus = True, delay = 0.05, format = 'png', 
                    subfolder = 'small_' + str(i),
                    focus_param = [200, 2], focus_roi = [0.2, 0.8, 0.2, 0.8], 
                    initial_focus_param = [8000, 30], focus_delay=0.17, 
                    focus_first_delay = 0.05, coarse_focus_first_delay = 2.5, 
                    mode = 'goto')
            raw_folder = self.scanner.parent_folder + '/small_' + str(i)
            color_folder = self.scanner.parent_folder + '/color_' + str(i)
            output_folder = self.scanner.parent_folder + '/results_' + str(i)
            if scan_only:
                self.search_thread[i] = threading.Thread(target = process_scan_only, 
                                                    args = (raw_folder, color_folder, output_folder, i))
            else:
                self.search_thread[i] = threading.Thread(target = start_search, 
                                                     args=(raw_folder, color_folder, output_folder, i), 
                                                     kwargs=kwargs)
            self.search_thread[i].start()
        
    def start_scan_revisit(self, magnification = '50x', focus_param = [200, 10], revisit = True):
        '''
        magnification: '20x' or '50x'
        focus_params: different params for 20x or 50x
        '''
        if magnification == '20x':
            self.stage.move_turret(3)
            focus_param = [300, 6]
        elif magnification == '50x':
            self.stage.move_turret(4)
            focus_param = [200, 10]
        time.sleep(1)
        self.stage.set_RGB_params(magnification)
        num = len(self.grid_points_list)
        self.analyze_revisit_thread = [[] for i in range(num)]
        for i in range(num):
            if self.search_thread[i] == []:
                continue
            self.search_thread[i].join()
            filepath = self.scanner.parent_folder + '/results_' + str(i) + '/revisit_pos.npy'
            if not (os.path.isfile(filepath) and revisit):
                continue
            revisit_pos = np.load(filepath)
            if len(revisit_pos) == 0:
                continue

            self.stage.goto_xyz(*revisit_pos[0])
            time.sleep(0.5)
            self.scanner.scan(revisit_pos, focus = True, delay = 0.07, format = 'png', 
                    subfolder='revisit_'+ str(i),
                    focus_param = focus_param, focus_roi = [0.2, 0.8, 0.2, 0.8], focus_delay=0.18, 
                    focus_first_delay = 0.05,initial_focus=False, mode = 'goto', focus_thre=1,
                    delay_R = 0.18, delay_B = 0.18)
            
            origin_folder = self.scanner.parent_folder + '/color_' + str(i)
            input_folder = self.scanner.parent_folder + '/revisit_' + str(i)
            output_folder = self.scanner.parent_folder + '/revisit_color_' + str(i)
            self.analyze_revisit_thread[i] = threading.Thread(target = analyze_revisit, 
                                                              args=(origin_folder, input_folder, output_folder, int(magnification[:-1])))
            self.analyze_revisit_thread[i].start()
            self.stage.move_xyz(z = 2600)
            time.sleep(0.5)
            
    def final_process(self):
        num = len(self.grid_points_list)
        for i in range(num):
            input_folder = self.scanner.parent_folder + '/revisit_color_' + str(i)
            output_folder = self.scanner.parent_folder + '/results_' + str(i)
            target_folder = 'E' + output_folder[1:]

            if not self.analyze_revisit_thread[i] == []:
                self.analyze_revisit_thread[i].join()
                merge_folder(input_folder, output_folder)
            
            os.makedirs(target_folder)
            copy_image(output_folder, target_folder)
            add_signature(target_folder)

        self.stage.move_turret(2)


    def combine_small(self):
        for i in range(len(self.grid_points_list)):
            combine(self.scanner.parent_folder + '/small_' + str(i), crop_w = [0.15, 0.75], compress = 8, scale = 0.25)    

    

#%%     
if __name__ == '__main__':
    camera = VieworksCamera()
    stage = Stage_MCX_E1000()
    light = RGBW_LED()

    temp = Scan_Platform(camera, stage, light)
    temp.initialize()

#%%
    temp.start_scan_large(2400, 2400, 3, 3)
#%%
    temp.start_scan_small(area = 1000, thickness_range = [0, 3], material = 'Graphene', revisit_magnification = '50x')