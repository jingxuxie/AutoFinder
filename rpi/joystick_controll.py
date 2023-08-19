from time import sleep
from general import Stage
from threading import Thread
import pygame

stage = Stage()
stage.initialize_joystick()
joystick_x = Thread(target = stage.move_x_joystick)
joystick_y = Thread(target = stage.move_y_joystick)
joystick_z = Thread(target = stage.move_z_joystick)
joystick_turret = Thread(target = stage.move_turret_joystick)

joystick_x.start()
joystick_y.start()
joystick_z.start()
joystick_turret.start()

joystick_x.join()
joystick_y.join()
joystick_z.join()
joystick_turret.join()
