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
filename = 'C:/Jingxu/platfrom_calibration_reverse.txt'
for i in range(pos_num):
    pos = position_list[i]
    stage.move_xyz(*pos)

    time.sleep(1)
    autofocus.focus(8000, 10, delay = 0.3, )

    with open(filename, 'a') as file:
        file.write(str(pos[0]) + ' ' + str(pos[1]) + ' ' + str(autofocus.height) + '\n')


# %%
pos_file = 'C:/Jingxu/platform_calibration.txt'

with open(pos_file) as f:
    lines = f.readlines()

x = np.linspace(0, 15000*5, 6)
y = np.linspace(0, 15000*5, 6)
X, Y = np.meshgrid(x, y)
Z = np.zeros((6, 6))

dir = 1
for i in range(6):
    for j in range(6):
        line = lines[i*6 + j]
        char = line.split()
        height = float(char[2])
        if dir == 1:
            Z[i, j] = height
        else:
            Z[i, -j-1] = height
    dir *= -1
# %%
