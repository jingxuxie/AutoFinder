#%%
import numpy as np
from autofinder.stages.stage_rpi import Stage_Rpi
import time
import cv2
from autofinder.Camera import Camera
from autofinder.autofocus import AutoFocus
from autofinder.findfocusplane import FindFocusPlane
import matplotlib.pyplot as plt
import os
from datetime import datetime
#%%
class Scanner():
    def __init__(self, camera, stage, plane_para = [0,0,1,0], magnification = '5x'):
        self.autofocus = AutoFocus(camera, stage)
        self.camera = self.autofocus.camera
        self.stage = self.autofocus.stage
        self.plane_para = plane_para
        self.magnification = magnification
        self.rotate_90 = False

        self.date = time.strftime("%m-%d-%Y")
        self.month = time.strftime('%m')
        self.day = time.strftime('%d')
        self.year = time.strftime('%Y')
        self.parent_folder = 'D:/JX_AutoFinder/'+self.year+'-'+self.month+'/'+self.day
        if not os.path.isdir(self.parent_folder):
            os.makedirs(self.parent_folder)

        self.save_count_dir = 1

        while os.path.isdir(self.parent_folder + '/' + str(self.save_count_dir)):
            self.save_count_dir += 1
        self.parent_folder += '/' + str(self.save_count_dir)
        os.makedirs(self.parent_folder+ '/raw')
        # os.makedirs(self.parent_folder+ '/temp')
        # os.makedirs(self.parent_folder+ '/results')
        self.position_filename = self.parent_folder + '/raw/position.txt'
        # self.save_folder += '/raw'

        
        self.save_count = 1
        
        self.index = '00_00-00_00'

        self.current_dir = os. getcwd().replace('\\', '/') + '/'
        self.bk_filename = self.current_dir + 'support_file/background/crop_x5.png'

        self.pos = np.zeros(3)


    def initialize(self):
        # ret_stage = self.stage.initialize()
        # ret_camera = self.camera.initialize()
        ret = self.autofocus.initialize()
        if ret: 
            return True
        return False


    def scan(self, position_list, speed_list = None, subfolder = None, focus = False, 
             compress = None, delay = 0, focus_delay = 0.1, format = 'jpg', mode = 'move', 
             fast = False, initial_focus_param = [8000, 10], focus_param = [800, 2], 
             focus_roi = [0, 1, 0, 1]):
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

        initial_focus = True

        for i in range(pos_num):
            pos = position_list[i]
            speed = speed_list[i]
            if mode == 'move':
                self.stage.move_xyz(*pos[:2], 0, *speed)
            elif mode == 'goto':
                self.stage.goto_xyz(*pos[:2], None, *speed)
                
            time.sleep(delay)
            self.save_img() if i > 0 else time.sleep(0.1)
            if focus:
                if initial_focus:
                    ret, self.img = self.autofocus.focus(*initial_focus_param, delay = focus_delay, roi = focus_roi)
                    if ret:
                        initial_focus = False

                _, self.img = self.autofocus.focus(*focus_param, fast = fast, delay = focus_delay, roi = focus_roi)
            else:
                self.capture()
            
            if mode == 'move':
                self.pos += np.array(pos)
            elif mode == 'goto':
                self.pos = np.array(pos)

        self.save_img()
         

    def capture(self):
        time.sleep(0.1)
        self.img = self.camera.get_frame()
        

    def save_img(self):
        if self.compress:
            width = int(self.img.shape[1] / self.compress)
            height = int(self.img.shape[0] / self.compress)
            self.img = cv2.resize(self.img, (width, height))
        filename = datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f")[:-4] + '000.' + self.format
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
    temp = Scanner(Camera(1), Stage_Rpi())
    temp.initialize()
    temp.scan(position)
    temp.stage.close()