#%%
from autofinder.cameras.vieworks import VieworksCamera
from autofinder.lights.RGB_LED import RGBW_LED
import numpy as np
import cv2
import time
#%%
class VieworksCameraRGB():
    def __init__(self, light):
        self.camera = VieworksCamera()
        self.light = light
        self.stop_movie = False
        self.mode = 'mono' # 'rgb' or 'mono'
        self.rgb_delay = 0.08
        self.mono_delay = 0.03
 
    def initialize(self):
        self.camera.initialize()
        self.shape = self.camera.last_frame.shape

    def get_frame(self):
        if self.mode == 'mono':
            self.last_frame = self.camera.last_frame
            time.sleep(self.mono_delay)
        elif self.mode == 'rgb':
            last_frame = np.zeros(list(self.shape) + [3], dtype = np.uint8)
            self.light.only_G()
            time.sleep(self.rgb_delay)
            last_frame[:, :, 1] = self.camera.last_frame
            self.light.only_R()
            time.sleep(self.rgb_delay)
            last_frame[:, :, 2] = self.camera.last_frame
            self.light.only_B()
            time.sleep(self.rgb_delay)
            last_frame[:, :, 0] = self.camera.last_frame

            self.last_frame = last_frame
        return self.last_frame

    def set_mode(self, mode):
        self.mode = mode
    
    def set_exposure_time(self, exposure_time:int):
        self.camera.set_exposure_time(exposure_time)

    def set_rgb_delay(self, delay:float):
        self.rgb_delay = delay

    def acquire_movie(self):
        while True:
            self.get_frame()
            if self.stop_movie:
                print('movie stopped')
                break

    def close_camera(self):
        pass

if __name__ == '__main__':
    light = RGBW_LED()
    cam = VieworksCameraRGB()

