#%%
import time
import numpy as np
from autofinder.stages.stage_mcx import Stage_MCX
from autofinder.stages.stage_e1000 import Stage_E1000

#%%
class Stage_MCX_E1000():
    def __init__(self):
        self.stage_mcx = Stage_MCX()
        self.stage_e1000 = Stage_E1000()

    def initialize(self):
        ret_mcx = self.stage_mcx.initialize()
        ret_e1000 = self.stage_e1000.initialize()

        if ret_mcx and ret_e1000:
            return True
        else:
            return False
        
    def move_xyz(self, x = 0, y = 0, z = 0, vx = 1, vy = 1, vz = 1):
        if not (x == 0 and y == 0):
            self.stage_mcx.move_xy(x = x, y = y)
            if z != 0:
                self.stage_e1000.move_z(z)
            self.stage_mcx.wait_stage()
        
        else:
            if z != 0:
                self.stage_e1000.move_z(z)

    def goto_xyz(self, x = 0, y = 0, z = None, vx = 1, vy = 1, vz = 1):
        if not (x == 0 and y == 0):
            self.stage_mcx.goto_xy(x = x, y = y)
            if z != None:
                self.stage_e1000.goto_z(z)
            self.stage_mcx.wait_stage()
        
        else:
            if z != None:
                self.stage_e1000.goto_z(z)

    def get_z(self):
        return self.stage_e1000.get_z()
    
    def get_turret(self):
        return self.stage_e1000.get_turret()

    def home(self):
        self.stage_mcx.home()
        self.stage_mcx.wait_stage()
    
    def move_turret(self, flag, sleep = 1):
        '''
        flag: 1 to 6
        '''
        self.stage_e1000.move_turret(flag, sleep)

    def set_RGB_params(self, magnification):
        '''
        magnification: '10x', '20x' or '50x'
        '''
        if magnification == '10x':
            self.stage_e1000.RGB_params = self.stage_e1000.RGB_params_10x
        elif magnification == '20x':
            self.stage_e1000.RGB_params = self.stage_e1000.RGB_params_20x
        elif magnification == '50x':
            self.stage_e1000.RGB_params = self.stage_e1000.RGB_params_50x


    def move_G_to_R(self):
        self.stage_e1000.move_G_to_R()
    
    def move_R_to_B(self):
        self.stage_e1000.move_R_to_B()

    def move_B_to_G(self):
        self.stage_e1000.move_B_to_G()

    def move_G_to_B(self):
        self.stage_e1000.move_G_to_B()

    def enable_joystick(self):
        self.stage_mcx.enable_joystick()

    def disable_joystick(self):
        self.stage_mcx.disable_joystick()

    def close(self):
        self.stage_mcx.close()
        self.stage_e1000.close()

#%%
if __name__ == '__main__':
    stage_mcx_e1000 = Stage_MCX_E1000()
    stage_mcx_e1000.initialize()
# %%
