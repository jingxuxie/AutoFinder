import os
import threading
from time import sleep
import RPi.GPIO as GPIO
#import pygame

os.environ["DISPLAY"] = ":0"

def get_keys():
    output_keys = ['x_pul', 'x_dir', 'y1_pul', 'y1_dir', 
                   'y2_pul', 'y2_dir', 'z_pul', 'z_dir',
                   'turret_pul', 'turret_dir']
    input_keys = ['x_limit_1', 'x_limit_2', 'y1_limit_1', 'y1_limit_2',
                  'y2_limit_1', 'y2_limit_2', 'z_limit_1', 'z_limit_2']
    return output_keys, input_keys

def get_pins():
    folder = os.getcwd()
    filename = '/home/pi/Documents'+ '/GPIOpins.txt'

    with open(filename) as f:
        lines = f.readlines()

    pins = []
    for i in range(9):
        line = lines[i].split()
        for j in range(2):
            pins.append(int(line[j]))
    
    # PUL, DIR, Limit_1, Limit_2
    output_keys, input_keys = get_keys()
    keys = output_keys + input_keys
    out = {}
    print(len(keys), len(pins))
    for i in range(len(keys)):
        key, pin = keys[i], pins[i]
        out[key] = pin

    print(out)
    return out, output_keys, input_keys

# get_pins()

class Stage():
    def __init__(self):
        
        self.pins, self.output_keys, self.input_keys = get_pins()
        self.limit_keys = self.input_keys

        self.output_pins = [self.pins[key] for key in self.output_keys]
        self.input_pins = [self.pins[key] for key in self.input_keys]

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.output_pins, GPIO.OUT)
        GPIO.setup(self.input_pins, GPIO.IN)

        # Limit_1, Limit_2
        self.limit_flag = {key:False for key in self.input_keys}

        self.stop_monitor_flag = False
        self.monitor_thread = threading.Thread(target = self.monitor_limit)
        self.monitor_thread.start()

        self.is_moving_flag = {item:False for item in ['x', 'y', 'z', 'turret']}

        

    def monitor_limit(self):
        while True:
            if self.stop_monitor_flag:
                break
            for key in self.limit_keys:
                pin = self.pins[key]
                if GPIO.input(pin):
                    self.limit_flag[key] = True
                    print(key, ' reached!')
                else:
                    self.limit_flag[key] = False
            sleep(0.005)

    def move_x(self, dir_flag, rot_num, interval):
        '''
        dir_flag: 0 or 1
        rot_num: int
        interval: float, in unit of second
        '''
        self.is_moving_flag['x'] = True
        GPIO.output(self.pins['x_dir'], dir_flag)
        sleep(0.001)
        for i in range(rot_num):
            if self.is_limit_x(dir_flag):
                break
            GPIO.output(self.pins['x_pul'], True)
            sleep(interval)
            GPIO.output(self.pins['x_pul'], False)
            sleep(interval)
        self.is_moving_flag['x'] = False


    def move_y(self, dir_flag, rot_num, interval):
        '''
        dir_flag: 0 or 1
        rot_num: int
        interval: float, in unit of second
        '''
        self.is_moving_flag['y'] = True
        GPIO.output([self.pins['y1_dir'], self.pins['y2_dir']], dir_flag)
        sleep(0.001)
        for i in range(rot_num):
            if self.is_limit_y(dir_flag):
                break
            GPIO.output([self.pins['y1_pul'], self.pins['y2_pul']], True)
            sleep(interval)
            GPIO.output([self.pins['y1_pul'], self.pins['y2_pul']], False)
            sleep(interval)
        self.is_moving_flag['y'] = False


    def move_z(self, dir_flag, rot_num, interval):
        '''
        dir_flag: 0 or 1
        rot_num: int
        interval: float, in unit of second
        '''
        self.is_moving_flag['z'] = True
        GPIO.output(self.pins['z_dir'], dir_flag)
        sleep(0.001)
        for i in range(rot_num):
            if self.is_limit_z(dir_flag):
                break
            GPIO.output(self.pins['z_pul'], True)
            sleep(interval)
            GPIO.output(self.pins['z_pul'], False)
            sleep(interval)
        self.is_moving_flag['z'] = False

    def move_turret(self, dir_flag, rot_num, interval):
        '''
        dir_flag: 0 or 1
        rot_num: int
        interval: float, in unit of second
        '''
        self.is_moving_flag['turret'] = True
        GPIO.output(self.pins['turret_dir'], dir_flag)
        sleep(0.001)
        for i in range(rot_num):
            if self.is_limit_z(dir_flag):
                break
            GPIO.output(self.pins['turret_pul'], True)
            sleep(interval)
            GPIO.output(self.pins['turret_pul'], False)
            sleep(interval)
        self.is_moving_flag['turret'] = False
        
    def initialize_joystick(self):
        import pygame
        global pygame
        pygame.init()
        pygame.joystick.init()
        try:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        except:
            print('Joystick not found')


    def move_x_joystick(self):
        GPIO.output(self.pins['x_dir'], 0)
        sleep(0.01)
        last_dir_flag = 0
        interval = 0.001
        while True:
            pygame.event.get()

            x_axis = self.joystick.get_axis(0)
            if abs(x_axis) < 0.1:
                sleep(0.01)
                continue
            interval = 0.0001 / abs(x_axis)**3

            dir_flag = 1 if x_axis < 0 else 0
            
            if self.is_limit_x(dir_flag):
                sleep(0.01)
                continue
            
            if dir_flag != GPIO.input(self.pins['x_dir']):
                GPIO.output(self.pins['x_dir'], dir_flag)
                sleep(0.01)
                last_dir_flag = dir_flag
            
            GPIO.output(self.pins['x_pul'], True)
            sleep(interval)
            GPIO.output(self.pins['x_pul'], False)
            sleep(interval)
            
    def move_y_joystick(self):
        GPIO.output([self.pins['y1_dir'], self.pins['y2_dir']], 0)
        sleep(0.01)
        last_dir_flag = 0
        interval = 0.0001
        
        while True:
            pygame.event.get()

            y_axis = self.joystick.get_axis(1)
            if abs(y_axis) < 0.1:
                sleep(0.1)
                continue
            interval = 0.0001 / abs(y_axis)**3

            dir_flag = 1 if y_axis < 0 else 0
            
            if self.is_limit_y(dir_flag):
                sleep(0.1)
                continue
            
            if dir_flag != GPIO.input(self.pins['y1_dir']):
                GPIO.output([self.pins['y1_dir'], self.pins['y2_dir']], dir_flag)
                sleep(0.01)
                last_dir_flag = dir_flag
            
            GPIO.output([self.pins['y1_pul'], self.pins['y2_pul']], True)
            sleep(interval)
            GPIO.output([self.pins['y1_pul'], self.pins['y2_pul']], False)
            sleep(interval)
        
    def move_z_joystick(self):
        GPIO.output(self.pins['z_dir'], 0)
        sleep(0.01)
        last_dir_flag = 0
        interval = 0.001
        while True:
            pygame.event.get()

            z_axis = self.joystick.get_axis(2)
            if abs(z_axis) < 0.1:
                sleep(0.01)
                continue
            interval = 0.0001 / abs(z_axis)**3

            dir_flag = 1 if z_axis < 0 else 0
            
            if self.is_limit_z(dir_flag):
                sleep(0.01)
                continue
            
            if dir_flag != GPIO.input(self.pins['z_dir']):
                GPIO.output(self.pins['z_dir'], dir_flag)
                sleep(0.01)
                last_dir_flag = dir_flag
            
            GPIO.output(self.pins['z_pul'], True)
            sleep(interval)
            GPIO.output(self.pins['z_pul'], False)
            sleep(interval)

    def move_turret_joystick(self):
        # GPIO.output(self.pins['turret_dir'], 0)
        sleep(0.01)
        last_dir_flag = 0
        interval = 0.001
        while True:
            pygame.event.get()

            turret_axis_0 = self.joystick.get_button(0)
            turret_axis_1 = self.joystick.get_button(2)

            if turret_axis_0 == 1 or turret_axis_1 == 1:
                dir_flag = 1 if turret_axis_0 == 0 else 0

                self.move_turret(dir_flag, 1400, 0.0005)
            
            sleep(0.01)
            
            # if last_dir_flag != dir_flag:
            #     GPIO.output(self.pins['z_dir'], dir_flag)
            #     sleep(0.01)
            #     last_dir_flag = dir_flag
            



    def is_limit_x(self, dir_flag):
        if (self.limit_flag['x_limit_1'] and dir_flag == 0) or \
           (self.limit_flag['x_limit_2'] and dir_flag == 1):
            return True
        else:
            return False

    def is_limit_y(self, dir_flag):
        if ((self.limit_flag['y1_limit_1'] or self.limit_flag['y2_limit_1']) and dir_flag == 0) or \
           ((self.limit_flag['y1_limit_2'] or self.limit_flag['y2_limit_2']) and dir_flag == 1):
            return True
        else:
            return False

    def is_limit_z(self, dir_flag):
        if (self.limit_flag['z_limit_1'] and dir_flag == 0) or \
           (self.limit_flag['z_limit_2'] and dir_flag == 1):
            return True
        else:
            return False

