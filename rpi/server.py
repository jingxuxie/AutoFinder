import socket
import time
import threading
import numpy as np
from general import Stage

def seperate_data(data):
    out = [[0, 0, 0] for i in range(4)]
    string = data.decode()
    chars = string.split()
    for i in range(4):
        for j in range(3):
            char = chars[i*3 + j]
            if j != 2:
                out[i][j] = int(char)
            else:
                out[i][j] = float(char)
    return out


stage = Stage()

HOST = "10.142.14.156"
PORT = 8000  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen()
    while True:
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            conn.sendall(b'Hello World')
            while True:
                data = conn.recv(1024)
                # print(data)
                if not data:
                    break
                # time.sleep(1)
                
                params = seperate_data(data)
                stage_x = threading.Thread(target = stage.move_x, args = params[0])
                stage_y = threading.Thread(target = stage.move_y, args = params[1])
                stage_z = threading.Thread(target = stage.move_z, args = params[2])
                stage_turret = threading.Thread(target = stage.move_turret, args = params[3])

                stage_x.start()
                stage_y.start()
                stage_z.start()
                stage_turret.start()

                stage_x.join()
                stage_y.join()
                stage_z.join()
                stage_turret.join()

                conn.sendall(b'success')

            
stage.stop_monitor_flag = True
stage.monitor_thread.join()        