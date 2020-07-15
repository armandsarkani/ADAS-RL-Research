import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import socket
import pickle
#import carla
import math
import time
import threading

d = None
conn_reset = False
class LaneDepartureData:
    location_x = 0
    location_y = 0
    right = 0
    left = 0
    right_x = 0
    right_y = 0
    right_lane_width = 0
    left_x = 0
    left_y = 0
    left_lane_width = 0
    acc_x = 0
    acc_y = 0
    acc_z = 0
def right_lane_departure_warning(conn, location_x, location_y, right, right_x, right_y, right_lane_width, acc_x, acc_y, acc_z, steer):
    threshold = -1
    acc_magnitude = math.sqrt((acc_x ** 2) + (acc_y ** 2) + (acc_z ** 2))
    '''if(acc_magnitude >= 1 and steer >= 0.1): # adjust threshold for acceleration
        threshold -= acc_magnitude * 0.5'''
    if(right == 0):
        return False
    if(acc_magnitude == 0): # if car is not moving, do not fire lane departure warning
        threshold = 0
    if(right == 1 and abs(location_x - right_x) <= 1): # if x are negligibly similar
        lane_width = right_lane_width
        lane_marking_x = right_x # constant x
        car_y = location_y
        if(car_y - right_y < 0):
            lane_marking_y = right_y - lane_width/2 # different y
            polarity = -1
        else:
            lane_marking_y = right_y + lane_width/2 # different y
            polarity = 1
        if(polarity == -1 and car_y - lane_marking_y >= threshold): #relative to lane marking check
            string = "Warning! Approaching right lane!, car y = " + str(car_y) + " LM y = " + str(lane_marking_y)
            conn.send(string.encode())
            return True
        elif(polarity == 1 and lane_marking_y - car_y >= threshold):
            string = "Warning! Approaching right lane!, car y = "+ str(car_y) + " LM y = " + str(lane_marking_y)
            conn.send(string.encode())
            return True
    elif(right == 1 and abs(location_y - right_y) <= 1): # if y are negligibly similar
        lane_width = right_lane_width
        lane_marking_y = right_y # constant y
        car_x = location_x
        if(car_x - right_x < 0):
            polarity = -1
            lane_marking_x = right_x - lane_width/2 # different y
        else:
            polarity = 1
            lane_marking_x = right_x + lane_width/2 # different y
        if(polarity == -1 and car_x - lane_marking_x >= threshold):
            string = "Warning! Approaching right lane!, car x = " + str(car_x) + " LM x = " + str(lane_marking_x)
            conn.send(string.encode())
            return True
        elif(polarity == 1 and lane_marking_x - car_x >= threshold):
            string = "Warning! Approaching right lane!, car x = " + str(car_x) + " LM x = " + str(lane_marking_x)
            conn.send(string.encode())
            return True
    string = "RightSafe"
    conn.send(string.encode())
    return False
def left_lane_departure_warning(conn, location_x, location_y, left, left_x, left_y, left_lane_width, acc_x, acc_y, acc_z, steer):
    threshold = -1
    acc_magnitude = math.sqrt((acc_x ** 2) + (acc_y ** 2) + (acc_z ** 2))
    '''if(acc_magnitude >= 1 and steer <= -0.1): # adjust threshold for acceleration
        threshold -= acc_magnitude * 0.5'''
    if(left == 0):
        return False
    if(acc_magnitude == 0): # if car is not moving, do not fire lane departure warning
        threshold = 0
    if(left == 1 and abs(location_x - left_x) <= 1): # if x are negligibly similar
        lane_width = left_lane_width
        lane_marking_x = left_x # constant x
        car_y = location_y
        if(car_y - left_y < 0):
            lane_marking_y = left_y - lane_width/2 # different y
            polarity = -1
        else:
            lane_marking_y = left_y + lane_width/2 # different y
            polarity = 1
        if(polarity == -1 and car_y - lane_marking_y >= threshold): #relative to lane marking check
            string = "Warning! Approaching left lane!, car y = " + str(car_y) + " LM y = " + str(lane_marking_y)
            conn.send(string.encode())
            return True
        elif(polarity == 1 and lane_marking_y - car_y >= threshold):
            string = "Warning! Approaching left lane!, car y = " + str(car_y) + " LM y = " + str(lane_marking_y)
            conn.send(string.encode())
            return True
    elif(left == 1 and abs(location_y- left_y) <= 1): # if y are negligibly similar
        lane_width = left_lane_width
        lane_marking_y = left_y # constant y
        car_x = location_x
        if(car_x - left_x < 0):
            polarity = -1
            lane_marking_x = left_x - lane_width/2 # different y
        else:
            polarity = 1
            lane_marking_x = left_x + lane_width/2 # different y
        if(polarity == -1 and car_x - lane_marking_x >= threshold):
            string = "Warning! Approaching left lane!, car x = " + str(car_x) + " LM x = " + str(lane_marking_x)
            conn.send(string.encode())
            return True
        elif(polarity == 1 and lane_marking_x - car_x >= threshold):
            string = "Warning! Approaching left lane!, car x = " + str(car_x) + " LM x = " + str(lane_marking_x)
            conn.send(string.encode())
            return True
    string = "LeftSafe"
    conn.send(string.encode())
    return False
def ThreadFunction(conn):
    global d
    global conn_reset
    while True:
        try:
            data = conn.recv(4096)
            d = pickle.loads(data)
        except ConnectionResetError:
            conn_reset = True
            print("Disconnected.")
            break
    
def main():
    global conn_reset
    device = sys.argv[1]
    if(device == "iMac"):
        HOST = '192.168.0.4' # iMac Pro
    elif(device == "MBPo"):
        HOST = '192.168.254.41' # 16-inch other
    elif(device == "MBP"):
        HOST = '192.168.0.78' # 16-inch
    else:
        HOST = 'localhost'
    PORT = 50007
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    print('Connected by', addr)
    thread = threading.Thread(target=ThreadFunction, args=(conn,))
    thread.start()
    while True:
        try:
            #data = conn.recv(4096)
            #d = pickle.loads(data)
            if(d is not None):
                right_lane_departure_warning(conn, d.location_x, d.location_y, d.right, d.right_x, d.right_y, d.right_lane_width, d.acc_x, d.acc_y, d.acc_z, d.steer)
                left_lane_departure_warning(conn, d.location_x, d.location_y, d.left, d.left_x, d.left_y, d.left_lane_width, d.acc_x, d.acc_y, d.acc_z, d.steer)
                time.sleep(1)
            if(conn_reset):
                s.close()
                conn.close()
                conn_reset = False
                main()
        except EOFError:
            thread.join()
            s.close()
            exit()
        except KeyboardInterrupt:
            s.close()
            conn.close()
            thread.join()
            exit()
    
    conn.close()
main()
