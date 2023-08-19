from paramiko.client import SSHClient, AutoAddPolicy
import time


client = SSHClient()
client.set_missing_host_key_policy(AutoAddPolicy)
client.connect('10.142.12.60', username = 'pi', password = 'raspberry')
stdin, stdout, stderr = client.exec_command('python3 Documents/led.py')
stdout.readlines()

from gpiozero import LED
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep

factory = PiGPIOFactory(host='169.254.168.219')
led = LED(19, pin_factory=factory)

for i in range(10):
    led.on()
    sleep(0.01)
    led.off()
    sleep(0.01)

#%%
import socket

HOST = "10.142.14.156"
PORT = 8000  # Port to listen on (non-privileged ports are > 1023)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
data = s.recv(1024)

#%%
start = time.time()
for i in range(10):
    s.sendall(b"0 0 0.0005 0 0 0.0005 0 0 0.1 0 1400 0.0005")
    data = s.recv(1024)
print(time.time() - start)
# %%
