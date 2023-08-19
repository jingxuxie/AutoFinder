#%%
import numpy as np
from autofinder.stages.stage_rpi import Stage_Rpi
import time
import cv2
from autofinder.Camera import Camera
import matplotlib.pyplot as plt

#%%
class AutoFocus():
    def __init__(self, camera, stage):
        self.camera = camera
        self.stage = stage

        self.gradient_mean = []
        self.height = 0
        self.error = False
        

    def initialize(self):
        ret_stage = self.stage.initialize()
        ret_camera = self.camera.initialize()
        # assert self.camera.get_frame().shape == (2160, 3840, 3)
        if not (ret_stage and ret_camera): 
            return False
        return True
    
    def focus(self, half_range, num:int, speed = 2, delay = 0.1, fast = False, **kwargs):
        skip_thre = 80

        step = half_range / num * 2
        self.stage.move_xyz(z = half_range, vz = speed)
        self.gradient_mean = []
        self.img_list = []
        if not fast:
            time.sleep(delay)
        self.capture()

        if np.max(self.img) < skip_thre: # if this is not on chip
            self.stage.move_xyz(z = - half_range, vz = speed)
            return False, self.img
            # self.stage.move_xyz(z = - 2 * half_range, vz = speed)
            # self.gradient_mean = []
            # self.img_list = []
            # if not fast:
            #     time.sleep(delay)
            # self.capture()
            # if np.max(self.img) < skip_thre:
            #     self.stage.move_xyz(z = half_range, vz = speed)
            #     return False, self.img
            # step = -step
            # half_range = -half_range

        for i in range(num):
            self.stage.move_xyz(z = - step, vz = speed)
            self.cal_clearness(**kwargs)
            self.capture(delay)
        self.cal_clearness(**kwargs)
        index_temp = np.argmax(self.gradient_mean)
        index = num - index_temp
        self.stage.move_xyz(z = index * step, vz = speed)
        self.height += (-half_range + index * step)
        print(index, self.height)
        print(self.gradient_mean)
        return True, self.img_list[index_temp]


    
    def coarse_focus(self):
        self.focus(4000, 10)

    def fine_focus(self):
        self.focus(400, 10)
        
    def capture(self, delay = 0.1):
        time.sleep(delay)
        self.img = self.camera.get_frame()
        self.img_list.append(self.img)

    def cal_clearness(self, metric = 'default', roi = [0, 1, 0, 1], 
                      first_process = False, brightness_min = 50):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        width = img.shape[1]
        height = img.shape[0]
        roi_w = [int(width * roi[0]), int(width * roi[1])]
        roi_h = [int(height * roi[2]), int(height * roi[3])]

        img = img[roi_h[0]: roi_h[1], roi_w[0]: roi_w[1]]

        if first_process:
            img_binary = cv2.inRange(img, brightness_min, 256)
            kernel_open = np.ones((50,50),np.uint8)
            img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel_open)

            contours, _ = cv2.findContours(img_open, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros(img.shape[:2], np.uint8)
            
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                if area > 1e5:
                    cv2.drawContours(mask, [contours[i]], -1, 255, -1)
                    cv2.dilate(mask, np.ones((200, 200), np.uint8))
            img[mask == 0] = 0

        if metric == 'default':
            img = cv2.medianBlur(img,5)
            gaussianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
            gaussianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)
            measure = np.mean(gaussianX**2 + gaussianY**2)

        elif metric == 'laplacian':
            laplacian = cv2.Laplacian(img,cv2.CV_64F)
            measure = np.var(np.abs(laplacian))

        else:
            print('unknow metric')

        self.gradient_mean.append(measure)

    def adaptive_focus(self):
        pass

    def close(self):
        self.stage.close()
        self.camera.close_camera()



#%%
if __name__ == '__main__':
    temp = AutoFocus(Camera(0), Stage_Rpi())
    temp.initialize()
# %%
    temp.coarse_focus()
# %%
    temp.fine_focus()

# %%
    temp.focus(2000, 10)

# %%
