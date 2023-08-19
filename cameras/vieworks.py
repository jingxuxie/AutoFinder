#%%
import numpy as np
import os
import cv2
from egrabber import *
import matplotlib.pyplot as plt
from threading import Thread
import time
#%%
def mono8_to_ndarray(rgb, w, h):
    data = ct.cast(rgb.get_address(), ct.POINTER(ct.c_ubyte * rgb.get_buffer_size())).contents
    c = 1
    return np.frombuffer(data, count=rgb.get_buffer_size(), dtype=np.uint8).reshape((h,w))
#%%
class VieworksCamera():
    def __init__(self, ROI = [0, 5120, 0, 5120]):
        self.initial_img_error = False
        self.initial_last_frame()
        self.camera_error = False

    def initial_last_frame(self):
        blank = np.zeros((512, 512, 3), dtype = np.uint8)
        self.last_frame = cv2.putText(blank, 'No Camera', (80, 280), 
                          cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 
                          5, cv2.LINE_AA)

    def initialize(self):
        gentl = EGenTL(Grablink())
        self.grabber = EGrabber(gentl)

        self.grabber.remote.set('Width', 5120)
        self.grabber.remote.set('Height', 5120)
        self.grabber.remote.set('DeviceTapGeometry', 'Geometry_1X10_1Y')

        self.grabber.device.set('CameraControlMethod', 'RG')
        self.grabber.device.set('ExposureTime', 1000)

        self.grabber.realloc_buffers(5)
        self.grabber.start()

        self.get_frame()
        Thread(target = self.acquire_movie).start()

        return True

    
    def get_frame(self):
        # Note: the buffer will be pushed back to the input queue automatically
        # when execution of the with-block is finished

        with Buffer(self.grabber, timeout=1000) as buffer:
            rgb = buffer.convert('Mono8')
            self.last_frame = mono8_to_ndarray(rgb, 5120, 5120).copy()
        # print(np.max(self.last_frame))
        return self.last_frame
    
    def set_exposure_time(self, exposure_time:int):
        self.grabber.device.set('ExposureTime', exposure_time)
    
    def acquire_movie(self):
        while True:
            self.get_frame()

    def close_camera(self):
        pass


#%%
if __name__ == '__main__':
    cam = VieworksCamera()
    cam.initialize()
    print(cam)
    # time.sleep(1)
    # frame = cam.get_frame()
    # Thread(target = cam.acquire_movie).start()
    

    # for i in range(100000):
    #     print(np.max(cam.last_frame))
        # time.sleep(0.1)
    #cam.close_camera()
    #print(np.max(frame))



# %%
