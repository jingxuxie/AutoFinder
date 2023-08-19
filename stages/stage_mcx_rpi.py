#%%
import time
import numpy as np
from autofinder.stages.stage_mcx import Stage_MCX
from autofinder.stages.stage_rpi import Stage_Rpi

#%%
class Stage_MCX_Rpi():
    def __init__(self):
        self.stage_mcx = Stage_MCX()
        self.stage_rpi = Stage_Rpi()

    def initialize(self):
        ret_mcx = self.stage_mcx.initialize()
        ret_rpi = self.stage_rpi.initialize()

        if ret_mcx and ret_rpi:
            return True
        else:
            return False
    
    def move_xyz(self, x = 0, y = 0, z = 0, vx = 1, vy = 1, vz = 1):
        if not (x == 0 and y == 0):
            self.stage_mcx.move_xy(x, y, vx, vy)
            if z != 0:
                self.stage_rpi.move_xyz(z, vz)
            self.stage_mcx.wait_stage()
        
        else:
            if z != 0:
                self.stage_rpi.move_xyz(z, vz)


    def home(self):
        self.stage_mcx.home()
        self.stage_mcx.wait_stage()

    def enable_joystick(self):
        self.stage_mcx.enable_joystick()
        self.stage_rpi.enable_joystick()

    def disable_joystick(self):
        self.stage_mcx.disable_joystick()
        self.stage_rpi.disable_joystick()

    def close(self):
        self.stage_mcx.close()
        self.stage_rpi.close()

#%%
if __name__ == '__main__':
    stage_mcx_rpi = Stage_MCX_Rpi()
    stage_mcx_rpi.initialize()