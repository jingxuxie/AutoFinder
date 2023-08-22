#%%
import serial
import time
import numpy as np

#%%
class Stage_E1000():
    def __init__(self):
        self.com = serial.Serial('COM3', timeout = 1)
        self.current_pos = np.zeros(3)
        self.RGB_params_10x = {'G_to_R': -100,
                               'R_to_B': -215,
                               'B_to_G': 315,
                               'G_to_B': -315}
        
        self.RGB_params_20x = {'G_to_R': -60,
                               'R_to_B': -120,
                               'B_to_G': 180,
                               'G_to_B': -180}
        
        self.RGB_params_50x = {'G_to_R': -13,
                               'R_to_B': -5,
                               'B_to_G': 18,
                               'G_to_B': -18}
        
        self.RGB_params = self.RGB_params_10x

    def initialize(self):
        self.com.write(b'SR0\r\n\r')
        ret = self.com.readline().decode()
        if 'SR0' in ret:
            print('Successfully connected to Nikon Eclipse E1000')
            return True
        else:
            print('Fail, no response from Nikon Eclipse E1000')
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

    def goto_z(self, z):
        char = self.decimal_to_hex(round(abs(z)))
        cmd = 'S0C' + char + '\r'
        self.com.write(cmd.encode())

    def get_z(self):
        self.com.read_all()
        self.com.write(b'SR0\r')
        ret = self.com.read(10)[3:-1].decode()
        try:
            current_z = int(ret, 16)
        except:
            time.sleep(1.5)
            self.com.read_all()
            self.com.write(b'SR0\r')
            ret = self.com.read(10)[3:-1].decode()
            current_z = int(ret, 16)

        return current_z


    def move_G_to_R(self):
        self.move_z(self.RGB_params['G_to_R'])
    
    def move_R_to_B(self):
        self.move_z(self.RGB_params['R_to_B'])

    def move_B_to_G(self):
        self.move_z(self.RGB_params['B_to_G'])

    def move_G_to_B(self):
        self.move_z(self.RGB_params['G_to_B'])

    def move_turret(self, flag, sleep = 1):
        '''
        flag: 1 to 6
        '''
        flag = str(flag)
        cmd = 'RD1' + flag + '\r'
        self.com.write(cmd.encode())
        time.sleep(sleep)

    def get_turret(self):
        self.com.read_all()
        self.com.write(b'RAR\r')
        ret = self.com.read(5)[3:-1].decode()
        return int(ret)
    
    def close(self):
        self.com.close()

#%%
if __name__ == '__main__':
    stage_e1000 = Stage_E1000()
    
# %%
