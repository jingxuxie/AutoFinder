#%%
import numpy as np
import time
from autofinder.autofocus import AutoFocus
from autofinder.auxiliary_func import generate_grid_points
from autofinder.Camera import Camera
from autofinder.stages.stage_rpi import Stage_Rpi


#%%
autofocus = AutoFocus(Camera(0), Stage_Rpi())
autofocus.initialize()
stage = autofocus.stage

#%%
position_list = generate_grid_points(x_step = 15000, y_step = 15000, x_num = 5, y_num = 5)
pos_num = len(position_list)

#%%
filename = 'C:/Jingxu/platfrom_calibration_15000.txt'
for i in range(pos_num):
    pos = position_list[i]
    stage.move_xyz(*pos)

    time.sleep(1)
    autofocus.focus(8000, 10, delay = 0.3, )

    with open(filename, 'a') as file:
        file.write(str(pos[0]) + ' ' + str(pos[1]) + ' ' + str(autofocus.height) + '\n')


# %%
