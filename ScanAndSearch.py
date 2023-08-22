#%%
import numpy as np
from autofinder.scan_platform import Scan_Platform
from autofinder.stages.stage_rpi import Stage_Rpi
from autofinder.lights import RGB_LED
from autofinder.stages.stage_mcx_e1000 import Stage_MCX_E1000
from autofinder.Camera import Camera
from autofinder.cameras.vieworks import VieworksCamera
import threading
import subprocess
import os
import time
from libsonyapi.camera import Camera_Wifi

class ScanAndSearch():
    def __init__(self, camera, stage, light, resultpath = 'E:/layer_search'):
        self.x_step = 2400
        self.y_step = 2400
        self.scan_area = '1 cm'
        self.flake_size = '20+ um'
        self.thickness_range_list = [[0, 10], [0, 0]]
        self.material = 'Graphene'
        self.revisit_magnification = '50x'
        self.scan_only = False
        self.revisit = True
        self.scan_platform = Scan_Platform(camera, stage, light)
        # ret = self.scan_platform.initialize()
        self.error = False # if ret else True
        # print('ScanAndSearch error:', self.error)

    def scan_and_search(self, **kwargs):
        # self.change_camera_setting()
        area = int(self.flake_size[:2])**2 * 4

        self.scan_platform.start_scan_large(self.x_step, self.y_step, x_num = 31, y_num = 21)
        self.change_camera_setting_small()
        self.scan_platform.start_scan_small(area = area, thickness_range = self.thickness_range_list, 
                                            material = self.material, scan_only = self.scan_only,
                                            revisit_magnification = self.revisit_magnification,
                                            **kwargs)
        self.change_camera_setting_revisit()
        self.scan_platform.start_scan_revisit(magnification = self.revisit_magnification, 
                                              revisit = self.revisit)
        self.scan_platform.final_process()

        self.scan_platform.stage.goto_xyz(70000, 0)
        self.scan_platform.stage.enable_joystick()

        print('Scan and search finished')


    def start_scan_and_search(self, **kwargs):
        main_thread = threading.Thread(target = self.scan_and_search, kwargs = kwargs)
        main_thread.start()


    def change_camera_setting_small(self):
        if self.material == 'hBN':
            self.scan_platform.camera.set_exposure_time(1000)
        else:
            self.scan_platform.camera.set_exposure_time(3000)
    
    def change_camera_setting_revisit(self):
        if self.material == 'hBN':
            self.scan_platform.camera.set_exposure_time(2000)
        else:
            self.scan_platform.camera.set_exposure_time(5000)

    '''
    def change_camera_setting(self):
        self.check_wifi_connection()
        camera_wifi = Camera_Wifi()
        while not camera_wifi.connected:
            self.check_wifi_connection()
            camera_wifi = Camera_Wifi()
            print('Connecting to the Camera')
        print('Camera connected')

        if self.material == 'hBN':
            camera_wifi.do("setIsoSpeedRate", param = '100')
            camera_wifi.do("setShutterSpeed", param = '1/4000')
        else:
            camera_wifi.do("setIsoSpeedRate", param = '125')
            camera_wifi.do("setShutterSpeed", param = '1/2000')
        time.sleep(2)
    '''


    def check_wifi_connection(self):
        while not b'DIRECT-sME1:ILCE-6400' in subprocess.check_output("netsh wlan show interfaces"):
            os.system(f'''cmd /c "netsh wlan connect name={"DIRECT-sME1:ILCE-6400"} interface={"WLAN"}"''')
            print('Connecting to the camera wifi ......')
            time.sleep(1)
        print('Camera Wifi connected')
    
        

#%%
if __name__ == '__main__':
    temp = ScanAndSearch(Camera(0), Stage_Rpi())
# %%
