#%%
import serial
import time
import numpy as np

#%%
class Stage_MCX():
    def __init__(self):
        self.com = serial.Serial('COM5', timeout = 5)
        self.current_pos = np.zeros(3)

    def initialize(self):
        self.com.write(b'0 0 76 52 setlimit ')
        self.com.write(b'1 1 setunit ')
        self.com.write(b'1 2 setunit ')
        
        self.com.write(b'ge ')
        reply = self.com.readline().decode().split()[0]
        if reply == '0':
            print('Successfully connected to MCX-2 eco controller')
            return True
        else:
            print('Fail, no response from MCX-2 controller')
            return False
    
    def move_xy(self, x = 0, y = 0, **kwargs):
        self.com.write((str(x) + ' ' + str(y) + ' rmove ').encode())
        self.wait_stage()

    def goto_xy(self, x = 0, y = 0, **kwargs):
        self.com.write((str(x) + ' ' + str(y) + ' move ').encode())
        self.wait_stage()
        pass

    def wait_stage(self):
        self.com.write(b'ge ')
        self.com.readline()


    def home(self):
        self.com.write(b'0 0 move ')
        self.wait_stage()
        self.current_pos = np.zeros(3)
        
    

    def enable_joystick(self):
        self.com.write(b'1 joystick ')


    def disable_joystick(self):
        self.com.write(b'0 joystick ')


    def close(self):
        self.com.close()




# %%
if __name__ == '__main__':
    temp = Stage_MCX()
    temp.initialize()
# %%
