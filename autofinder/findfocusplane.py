#%%
import numpy as np
from autofinder.stages.stage_rpi import Stage_Rpi
import time
import cv2
from autofinder.Camera import Camera
from autofinder.autofocus import AutoFocus
import matplotlib.pyplot as plt
from autofinder.plane_formula import Get_Plane_Para, Get_Z_Position


#%%
class FindFocusPlane():
    def __init__(self, camera, stage, focus_method = 'Single point', size = '10 mm'):
        self.autofocus = AutoFocus(camera, stage)
        self.stage = self.autofocus.stage
        self.para = [0, 0, 1, 0]
        self.focus_method = focus_method
        self.size = int(size[:-3])

    def initialize(self):
        ret = self.autofocus.initialize()
        if ret: 
            return True
        return False

    def focus(self):
        self.autofocus.focus(8000, 10, speed = 1)
        # self.autofocus.focus(800, 10, speed = 1)

    def find_plane(self):
        if self.focus_method == 'Single point':
            center = max(0, self.size * 500)
            self.stage.move_xyz(center, center, 0)
            self.focus()
            self.stage.move_xyz(-center, -center, 0)
        else:
            self.plane_points = []
            
            start = max(0, self.size * 100)
            step = self.size * 800
            start_point = [start, start]
            self.stage.move_xyz(*start_point, 0)

            self.autofocus.height = 0

            focus_point_1 = [0, 0]
            self.focus()
            self.plane_points.append([focus_point_1[0] + start_point[0],
                                      focus_point_1[1] + start_point[1],
                                      self.autofocus.height])

            focus_point_2 = [0, step]
            self.stage.move_xyz(*focus_point_2, 0)
            self.focus()
            self.plane_points.append([focus_point_2[0] + start_point[0],
                                      focus_point_2[1] + start_point[1],
                                      self.autofocus.height])

            focus_point_3 = [step, 0]
            self.stage.move_xyz(*focus_point_3, 0)
            self.focus()
            self.plane_points.append([focus_point_3[0] + start_point[0],
                                      focus_point_2[1] + start_point[1],
                                      self.autofocus.height])

            self.para = Get_Plane_Para(*self.plane_points)
            height = Get_Z_Position(0, 0, self.para)
            pos_temp = [-start_point[0] - focus_point_3[0],
                        -start_point[1] - focus_point_2[1],
                        height - self.plane_points[2][2]]
            self.stage.move_xyz(*pos_temp)


# %%
if __name__ == '__main__':
    temp = FindFocusPlane(Camera(0), Stage_Rpi(), 1)
    temp.initialize()

    temp.stage.move_xyz(1000)
    temp.stage.move_xyz(-1000)
    time.sleep(0.3)
    img = temp.autofocus.camera.get_frame()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img)
# %%
def capture():
    start = time.time()
    for i in range(50):
        img_temp = temp.autofocus.camera.get_frame()
        gray_1 = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
        time.sleep(0.05)
        
        img = temp.autofocus.camera.get_frame()
        gray_2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        d_frame = cv2.absdiff(gray_1, gray_2)
        
        if np.sum(d_frame[d_frame > 10]) < 10000:
            break
        if time.time() - start > 0.5:
            break
    return img
    # plt.figure()
    # plt.imshow(img)
# %%
