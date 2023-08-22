#%%
import numpy as np
from autofinder.stages.stage_rpi import Stage_Rpi
from autofinder.stages.stage_mcx_e1000 import Stage_MCX_E1000
import time
import cv2
from autofinder.Camera import Camera
from autofinder.cameras.vieworks import VieworksCamera
from autofinder.autofocus import AutoFocus
from autofinder.findfocusplane import FindFocusPlane
import matplotlib.pyplot as plt
import os
from datetime import datetime
from autofinder.lights.RGB_LED import RGBW_LED
from autofinder.auxiliary_func import generate_grid_points

#%%
class Scanner():
    def __init__(self, camera, stage, light = None, plane_para = [0,0,1,0], magnification = '5x'):
        self.autofocus = AutoFocus(camera, stage)
        self.camera = self.autofocus.camera
        self.stage = self.autofocus.stage
        self.light = light
        self.plane_para = plane_para
        self.magnification = magnification
        self.rotate_90 = False

        self.date = time.strftime("%m-%d-%Y")
        self.month = time.strftime('%m')
        self.day = time.strftime('%d')
        self.year = time.strftime('%Y')
        self.parent_folder = 'D:/0_JX_AutoFinder/' + self.year + '-' + self.month + '/' + self.day
        if not os.path.isdir(self.parent_folder):
            os.makedirs(self.parent_folder)

        self.save_count_dir = 1

        while os.path.isdir(self.parent_folder + '/' + str(self.save_count_dir)):
            self.save_count_dir += 1
        self.parent_folder += '/' + str(self.save_count_dir)
        os.makedirs(self.parent_folder+ '/raw')
        self.position_filename = self.parent_folder + '/raw/position.txt'
        
        self.pos = np.zeros(3)


    def initialize(self):
        # ret_stage = self.stage.initialize()
        # ret_camera = self.camera.initialize()
        ret = self.autofocus.initialize()
        if ret: 
            return True
        return False


    def scan(self, position_list, speed_list = None, subfolder = None, focus = False, 
             compress = None, delay = 0.03, focus_first_delay = 0.5, focus_delay = 0.1, 
             coarse_focus_first_delay = 2.5, format = 'jpg', mode = 'move', 
             fast = False, initial_focus_param = [2000, 10], focus_param = [800, 2], 
             focus_roi = [0, 1, 0, 1], initial_focus = True, focus_speed = 1, 
             RGB_color = True, focus_thre = 20, delay_R = 0.1, delay_B = 0.15):
        '''
        position_list: list or array of [x, y, z] positions
        speed_list: list or array of [vx, vy, vz] speed
        focus_roi: [crop_w, crop_h]
        '''
        if subfolder:
            self.save_folder = self.parent_folder + '/' + subfolder
            if not os.path.isdir(self.save_folder):
                os.makedirs(self.save_folder)
        else:
            self.save_folder = self.parent_folder + '/raw'
        self.position_filename = self.save_folder + '/position.txt'

        self.compress = compress
        self.format = format

        pos_num = len(position_list)
        if not speed_list:
            speed_list = [[1, 1, 1] for i in range(pos_num)]
        
        self.light.off_all()
        initial_height = self.stage.get_z()
        self.autofocus.height = 0
        self.height = 0
        for i in range(pos_num):
            pos = position_list[i]
            speed = speed_list[i]
            self.light.only_G()
            if mode == 'move':
                if i == 0:
                    self.stage.move_xyz(*pos[:2], 0, *speed)
                else:
                    self.stage.move_B_to_G() if RGB_color else None
                    self.stage.move_xyz(*pos[:2], 0, *speed)
            elif mode == 'goto':
                if i == 0:
                    self.stage.goto_xyz(*pos[:2], None, *speed)
                else:
                    self.stage.move_B_to_G() if RGB_color else None
                    if len(pos) == 2:
                        self.stage.goto_xyz(*pos[:2], None, *speed)
                    elif len(pos) == 3:
                        self.stage.goto_xyz(*pos, *speed)
                        initial_height = pos[2]
                
            time.sleep(delay)
            self.save_img_RGB('B') if i > 0 else time.sleep(0.1)
            if focus:
                if initial_focus:
                    ret, self.img = self.autofocus.focus(*initial_focus_param, 
                                                         first_delay = coarse_focus_first_delay, delay = focus_delay, 
                                                         roi = focus_roi, speed = focus_speed, 
                                                         first_process = True)
                    time.sleep(coarse_focus_first_delay)
                    if ret:
                        initial_focus = False

                _, self.img = self.autofocus.focus(*focus_param, fast = fast, 
                                                   first_delay = focus_first_delay, delay = focus_delay, 
                                                   roi = focus_roi, speed = focus_speed, skip_thre = focus_thre)
                self.img = self.img[1060:4060, 1060:4060]
                self.height = initial_height + self.autofocus.height
                self.capture_RGB(delay_R, delay_B)
            else:
                self.capture()
            
            if mode == 'move':
                self.pos += np.array(list(pos[:2]) + [self.height])
            elif mode == 'goto':
                self.pos = np.array(list(pos[:2]) + [self.height])

        time.sleep(0.15)
        self.save_img_RGB('G')
        self.stage.move_B_to_G() if RGB_color else None
        self.light.off_all()
         

    def capture(self):
        time.sleep(0.1)
        self.img = self.camera.last_frame[1060:4060, 1060:4060]


    def save_img(self):
        if self.compress:
            width = int(self.img.shape[1] / self.compress)
            height = int(self.img.shape[0] / self.compress)
            self.img = cv2.resize(self.img, (width, height))
        filename = datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f")[:-4] + '000.' + self.format
        cv2.imwrite(self.save_folder + '/' + filename, self.img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        with open(self.position_filename, 'a') as file:
            file.write(filename + ' ' + ' '.join(map(str, self.pos)) + '\n')


    def capture_RGB(self, delay_R, delay_B):
        self.light.only_R()
        self.stage.move_G_to_R()
        self.save_img_RGB('G')
        time.sleep(delay_R)
        self.img = self.camera.last_frame[1060:4060, 1060:4060]

        self.light.only_B()
        self.stage.move_R_to_B()
        self.save_img_RGB('R')
        time.sleep(delay_B)
        self.img = self.camera.last_frame[1060:4060, 1060:4060]

    
    def save_img_RGB(self, color):
        if self.compress:
            width = int(self.img.shape[1] / self.compress)
            height = int(self.img.shape[0] / self.compress)
            self.img = cv2.resize(self.img, (width, height))
        filename = datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f")[:-4] + color + '000.' + self.format
        cv2.imwrite(self.save_folder + '/' + filename, self.img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        with open(self.position_filename, 'a') as file:
            file.write(filename + ' ' + ' '.join(map(str, self.pos)) + '\n')

        


    def get_background(self):
        self.background = cv2.imread(self.bk_filename)
        self.background_norm = np.zeros(3)
        for i in range(3):
            self.background_norm[i] = np.mean(self.background[:,:,i])
            print(self.background_norm[i])

    def close(self):
        self.autofocus.close()


#%%
if __name__ == '__main__':
    camera = VieworksCamera()
    stage = Stage_MCX_E1000()
    light = RGBW_LED()
    temp = Scanner(camera, stage, light)
    temp.initialize()

    position_list = generate_grid_points(1200, 1200, 16, 16)
    position_list = generate_grid_points(2400, 2400, 31, 21)

#%%
    temp.scan(position_list, subfolder = 'large', compress = 8, 
              delay = 0.1, RGB_color=False)
#%%
    temp.scan(position_list, focus = True, delay = 0.05, format = 'png', 
              focus_param = [200, 2], focus_roi = [0.2, 0.8, 0.2, 0.8], 
              initial_focus_param = [8000, 30], focus_delay=0.17, 
              focus_first_delay = 0.05, coarse_focus_first_delay = 2.5)
#%%
    temp.scan(position_list, focus = True, delay = 0.03, format = 'png', 
              focus_param = [200, 2], focus_roi = [0.2, 0.8, 0.2, 0.8], 
              initial_focus_param = [4000, 20], focus_delay=0.02)

#%%
    from autofinder.analyze_large_scan import analyze_large_scan
    grid_points_list = analyze_large_scan(temp.save_folder)
    temp.camera.set_exposure_time(3000)
    num = len(grid_points_list)
    for i in range(num):
        points = grid_points_list[i]
        temp.stage.goto_xyz(*points[0])
        temp.scan(points, focus = True, delay = 0.05, format = 'png', subfolder = 'small_' + str(i),
                focus_param = [200, 2], focus_roi = [0.2, 0.8, 0.2, 0.8], 
                initial_focus_param = [8000, 30], focus_delay=0.17, 
                focus_first_delay = 0.05, coarse_focus_first_delay = 2.5, mode = 'goto')
    # temp = Scanner(Camera(1), Stage_Rpi())
    # temp.initialize()
    # temp.scan(position)
    # temp.stage.close()
# %%
    from autofinder.auxiliary_func import generate_revisit_list
    input_folder = 'D:/JX_AutoFinder/2023-08/16/10/color_shift_0'
    output_folder = 'D:/JX_AutoFinder/2023-08/16/10/results_0'
    pos, _ = generate_revisit_list(input_folder, output_folder)
    temp.stage.set_RGB_params('50x')
    temp.scan(pos, focus = True, delay = 0.07, format = 'png', subfolder='50x',
              focus_param = [200, 10], focus_roi = [0.2, 0.8, 0.2, 0.8], focus_delay=0.18, 
              focus_first_delay = 0.05,initial_focus=False, mode = 'goto', focus_thre=1)
