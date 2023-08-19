#%%
import numpy as np
import time
from autofinder.scan import Scanner
from autofinder.analyze_large_scan import analyze_large_scan, combine
from autofinder.stages.stage_rpi import Stage_Rpi
from autofinder.stages.stage_mcx_e1000 import Stage_MCX_E1000
import time
import cv2
from autofinder.Camera import Camera
from autofinder.cameras.vieworks import VieworksCamera
import matplotlib.pyplot as plt
import threading
import os
from shutil import copy2
# from autofinder.search_layers import start_search
from autofinder.auxiliary_func import crop_quarter_of_20, add_signature, merge_folder, copy_image, generate_grid_points
#%%
class Scan_Platform():
    def __init__(self, camera, stage):
        self.scanner = Scanner(camera, stage)
        self.stage = self.scanner.stage
        self.autofocus = self.scanner.autofocus

    def initialize(self):
        ret = self.scanner.initialize()
        if ret:
            return True
        return False
    
    def start_scan_large(self, x_step = 3000, y_step = 3000, x_num = 25, y_num = 25):
        position = generate_grid_points(x_step, y_step, x_num, y_num)
        self.scanner.scan(position, subfolder = 'large', compress = 8, delay = 0.1)
        self.grid_points_list = analyze_large_scan(self.scanner.save_folder)

    def start_scan_small(self, fast = False, scan_only = False, **kwargs):
        self.stage.move_turret(1)
        num = len(self.grid_points_list)
        self.search_thread = [[] for i in range(num)]
        for i in range(num):
            points = self.grid_points_list[i]
            self.stage.goto_xyz(*points[0])
            self.scanner.scan(points, subfolder = 'small_' + str(i), focus = True, mode = 'goto', fast = fast,
                              initial_focus_param=[32000, 20])
            if scan_only:
                continue
            input_folder = self.scanner.parent_folder + '/small_' + str(i)
            output_folder = self.scanner.parent_folder + '/results_' + str(i)
            self.search_thread[i] = threading.Thread(target = start_search, args=(input_folder, output_folder, i), kwargs=kwargs)
            self.search_thread[i].start()
        
    def start_scan_revisit(self):
        num = len(self.grid_points_list)
        self.crop_thread = [[] for i in range(num)]
        for i in range(num):
            if self.search_thread[i] == []:
                continue
            self.search_thread[i].join()
            filepath = self.scanner.parent_folder + '/results_' + str(i) + '/revisit_pos.npy'
            if not os.path.isfile(filepath):
                continue
            revisit_pos = np.load(filepath)
            if len(revisit_pos) == 0:
                continue

            self.stage.goto_xyz(*revisit_pos[0])
            self.autofocus.focus(32000, 20)
            self.stage.move_turret(1)
            self.stage.move_xyz(z = -2000)
            self.scanner.scan(revisit_pos, subfolder = '20x_' + str(i), focus = True, mode = 'goto',
                              initial_focus_param = [2000, 4],focus_param = [2000, 8], focus_delay = 0.4,
                              focus_roi = [0.2, 0.7, 0.3, 0.8])
            self.stage.move_turret(0)
            
            origin_folder = self.scanner.parent_folder + '/small_' + str(i)
            input_folder = self.scanner.parent_folder + '/20x_' + str(i)
            output_folder = self.scanner.parent_folder + '/20x_crop_' + str(i)
            copy2(self.scanner.parent_folder + '/small_' + str(i) + '/combine.jpg', input_folder)
            self.crop_thread[i] = threading.Thread(target = crop_quarter_of_20, args=(origin_folder, input_folder, output_folder))
            self.crop_thread[i].start()
            
    def final_process(self):
        num = len(self.grid_points_list)
        for i in range(num):
            input_folder = self.scanner.parent_folder + '/20x_crop_' + str(i)
            output_folder = self.scanner.parent_folder + '/results_' + str(i)
            target_folder = 'E' + output_folder[1:]

            if not self.crop_thread[i] == []:
                self.crop_thread[i].join()
                merge_folder(input_folder, output_folder)
            
            os.makedirs(target_folder)
            copy_image(output_folder, target_folder)
            add_signature(target_folder)

        self.stage.move_turret(0)


    def combine_small(self):
        for i in range(len(self.grid_points_list)):
            combine(self.scanner.parent_folder + '/small_' + str(i), crop_w = [0.15, 0.75], compress = 8, scale = 0.25)    

    

#%%     
if __name__ == '__main__':
    temp = Scan_Platform(VieworksCamera(), Stage_MCX_E1000())
    temp.initialize()

# %%
