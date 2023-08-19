import os
import threading
from time import sleep
from general import Stage
import RPi.GPIO as GPIO

stage = Stage()

stage_x = threading.Thread(target = stage.move_x, args = (0, 100, 0.0005))
stage_y = threading.Thread(target = stage.move_y, args = (0, 200, 0.0005))

stage_x.start()
stage_y.start()

stage_x.join()

stage_x1 = threading.Thread(target = stage.move_x, args = (0, 100, 0.0005))
stage_x1.start()
stage_x1.join()

stage_y.join()

stage.stop_monitor_flag = True
stage.monitor_thread.join()

GPIO.cleanup()
print('success')