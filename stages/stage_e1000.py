#%%
import serial
import time
import numpy as np

#%%
class Stage_E1000():
    def __init__(self):
        self.com = serial.Serial('COM3', timeout = 5)
        self.current_pos = np.zeros(3)

    def initialize(self):
        self.com.write(b'SR0\r\n\r')
        ret = self.com.readline().decode()
        if 'SR0' in ret:
            return True
        else:
            return False

    def decimal_to_hex(self, decimal: int):
        out =  hex(decimal)[2:].zfill(6).upper()
        return out

    def move_z(self, z):
        char = self.decimal_to_hex(round(abs(z)))

        if z == 0:
            return
        elif z > 0:
            cmd = 'S0D' + char + '\r'
        else:
            cmd = 'S0E' + char + '\r'
            
        self.com.write(cmd.encode())

    def move_G_to_R(self):
        self.move_z(-100)
    
    def move_R_to_B(self):
        self.move_z(-215)

    def move_B_to_G(self):
        self.move_z(315)

    def move_G_to_B(self):
        self.move_z(-315)

    def move_turret(self, flag, sleep = 1):
        '''
        flag: 1 to 6
        '''
        flag = str(flag)
        cmd = 'RD1' + flag + '\r'
        self.com.write(cmd.encode())
        time.sleep(sleep)
    
    def close(self):
        self.com.close()

#%%
if __name__ == '__main__':
    stage_e1000 = Stage_E1000()
    
# %%
