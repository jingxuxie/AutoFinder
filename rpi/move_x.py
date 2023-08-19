import os
import threading
from time import sleep
from general import Stage
import RPi.GPIO as GPIO

stage = Stage()

stage_x = threading.Thread(target = stage.move_x, args = (1, 40, 0.0001))
stage_x.start()
#while True:
#    if not stage.is_moving_flag['x']:
#        break
#    else:
#        sleep(0.005)
stage_x.join()
stage.stop_monitor_flag = True
stage.monitor_thread.join()
GPIO.cleanup()
print('success')