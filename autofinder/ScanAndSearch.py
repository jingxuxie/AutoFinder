#%%
import numpy as np
from autofinder.scan_platform import Scan_Platform
from autofinder.stages.stage_rpi import Stage_Rpi
from autofinder.Camera import Camera
import threading
import subprocess
import os
import time
from libsonyapi.camera import Camera_Wifi

class ScanAndSearch():
    def __init__(self, camera, stage, resultpath = 'E:/layer_search'):
        self.x_step = 2000
        self.y_step = 2000
        self.scan_area = '1 cm'
        self.flake_size = '20+ um'
        self.thickness_min = '0 nm'
        self.thickness_max = '50 nm'
        self.material = 'Graphene'
        self.resultpath = resultpath
        self.scan_platform = Scan_Platform(camera, stage)
        ret = self.scan_platform.initialize()
        self.error = False if ret else True
        print('ScanAndSearch error:', self.error)

    def scan_and_search(self, **kwargs):
        self.change_camera_setting()

        x_num = round(int(self.scan_area[0]) * 10000 / self.x_step) + 2
        y_num = round(int(self.scan_area[0]) * 10000 / self.y_step) + 2
        area = int(self.flake_size[:2])**2 * 4
        thickness_range = [float(self.thickness_min[:-3]), float(self.thickness_max[:-3])]
        
        self.scan_platform.stage.disable_joystick()
        self.scan_platform.stage.home()
        self.scan_platform.stage.move_xyz(3000, 3000)
        self.scan_platform.stage.current_pos = np.zeros(3)
        self.scan_platform.start_scan_large(self.x_step, self.y_step, x_num, y_num)
        self.scan_platform.start_scan_small(area = area, thickness_range = thickness_range, 
                                            material = self.material, **kwargs)
        self.scan_platform.start_scan_revisit()
        self.scan_platform.final_process()

        self.scan_platform.stage.move_xyz(100000, 100000)
        self.scan_platform.stage.enable_joystick()

        print('Scan and search finished')


    def start_scan_and_search(self, **kwargs):
        main_thread = threading.Thread(target = self.scan_and_search, kwargs = kwargs)
        main_thread.start()


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
