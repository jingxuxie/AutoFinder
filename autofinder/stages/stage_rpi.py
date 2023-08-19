#%%
import socket
import time
import numpy as np
#%%
class Stage_Rpi():

    def __init__(self):
        self.scale = [0.158, 0.158, 0.16] # 1000 um = 300 rotation steps
        self.default_interval = 5e-4 # 1000 rotation steps per second
        self.current_pos = np.zeros(3)

    def initialize(self):
        host = "192.168.1.128"
        port = 8000

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.settimeout(20)
        try:
            self.s.connect((host, port))
            reply = self.s.recv(1024)

            if reply == b'Hello World':
                print('Successfully connected to RPI server')
                return True

        except Exception as e:
            print(e)
            print('Fail, RPI server has no respond')
            return False


    def pos2str(self, position, speed) -> str:
        position[1] = -position[1]
        cmd = ''
        for i in range(3):
            pos = position[i]
            sign = 0 if pos <= 0 else 1
            rot = round(abs(pos * self.scale[i]))
            v = self.default_interval / speed[i]
            cmd += str(sign) + ' ' + str(rot) + ' ' + str(v) + ' '
        cmd += '0 0 1e-3'

        return cmd


    def move_xyz(self, x = 0, y = 0, z = 0, vx = 1, vy = 1, vz = 1):
        cmd = self.pos2str([x, y, z], [vx, vy, vz])
        self.s.sendall(cmd.encode())
        reply = self.s.recv(1024)
        if reply == b'success':
            self.current_pos += np.array([x, y, z], dtype = np.float64)
            return True
        else:
            print('move stage error')
            return False
        
    def goto_xyz(self, x = None, y = None, z = None, vx = 1, vy = 1, vz = 1):
        delta_x = x - self.current_pos[0] if x != None else 0
        delta_y = y - self.current_pos[1] if y != None else 0
        delta_z = z - self.current_pos[2] if z != None else 0
        ret = self.move_xyz(delta_x, delta_y, delta_z, vx, vy, vz)
        if ret:
            return True
        else:
            return False

    def move_turret(self, dir):
        '''
        dir: 0 or 1
        '''
        dir = str(dir)
        cmd = '0 0 1e-3 0 0 1e-3 0 0 1e-3 ' + dir + ' 1400 5e-4'
        self.s.sendall(cmd.encode())
        reply = self.s.recv(1024)
        if reply == b'success':
            return True
        else:
            return False

    def enable_joystick(self):
        cmd = 'enable_joystick'
        self.s.sendall(cmd.encode())
        reply = self.s.recv(1024)
        if reply == b'success':
            print('Joystick enabled')
            return True
        else:
            return False

    def disable_joystick(self):
        cmd = 'disable_joystick'
        self.s.sendall(cmd.encode())
        reply = self.s.recv(1024)
        if reply == b'success':
            print('Joystick disabled')
            return True
        else:
            return False

    def home(self):
        self.move_xyz(-1e6, -1e6, 0, 2, 2, 2)
        self.current_pos = np.zeros(3)


    def close(self):
        self.s.close()

        

    

    

# %%
