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
import RPi.GPIO as GPIO
import pickle
#import carla
import time
import threading

d = None

# Buttons:
BTN_G = 22 # GPIO 25
BTN_R = 12 # GPIO 18
BTN_Y = 13 # GPIO 27
BTN_B = 15 # GPIO 22

#LEDs:
LED_G = 29 # GPIO 5
LED_R = 31 # GPIO 6
LED_Y = 32 # GPIO 12
LED_B = 33 # GPIO 13

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
def GPIORed():
    GPIO.output(BTN_R, True)
    GPIO.output([LED_R], GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output([LED_R], GPIO.LOW)
    time.sleep(0.5)
    GPIO.output(BTN_R, False)
def GPIOBlue():
    GPIO.output(BTN_B, True)
    GPIO.output([LED_B], GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output([LED_B], GPIO.LOW)
    time.sleep(0.5)
    GPIO.output(BTN_B, False)
def GPIOGreen():
    GPIO.output(BTN_G, True)
    GPIO.output([LED_G], GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output([LED_G], GPIO.LOW)
    time.sleep(0.5)
    GPIO.output(BTN_G, False)
def right_lane_departure_warning(conn, location_x, location_y, right, right_x, right_y, right_lane_width): #red
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
        if(polarity == -1 and car_y - lane_marking_y >= -1): #relative to lane marking check
            string = "Warning! Approaching right lane!, car y = " + str(car_y) + " LM y = " + str(lane_marking_y)
            conn.send(string.encode())
            GPIORed()
            return True
        elif(polarity == 1 and lane_marking_y - car_y >= -1):
            string = "Warning! Approaching right lane!, car y = "+ str(car_y) + " LM y = " + str(lane_marking_y)
            conn.send(string.encode())
            GPIORed()
            return True
    elif(right == 1 and abs(location_y - right_y) <= 1): # if x are negligibly similar
        lane_width = right_lane_width
        lane_marking_y = right_y # constant y
        car_x = location_x
        if(car_x - right_x < 0):
            polarity = -1
            lane_marking_x = right_x - lane_width/2 # different y
        else:
            polarity = 1
            lane_marking_x = right_x + lane_width/2 # different y
        if(polarity == -1 and car_x - lane_marking_x >= -1):
            string = "Warning! Approaching right lane!, car x = " + str(car_x) + " LM x = " + str(lane_marking_x)
            conn.send(string.encode())
            GPIORed()
            return True
        elif(polarity == 1 and lane_marking_x - car_x >= -1):
            string = "Warning! Approaching right lane!, car x = " + str(car_x) + " LM x = " + str(lane_marking_x)
            conn.send(string.encode())
            GPIORed()
            return True
    return False
def left_lane_departure_warning(conn, location_x, location_y, left, left_x, left_y, left_lane_width): #blue
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
        if(polarity == -1 and car_y - lane_marking_y >= -1): #relative to lane marking check
            string = "Warning! Approaching left lane!, car y = " + str(car_y) + " LM y = " + str(lane_marking_y)
            conn.send(string.encode())
            GPIOBlue()
            return True
        elif(polarity == 1 and lane_marking_y - car_y >= -1):
            string = "Warning! Approaching left lane!, car y = " + str(car_y) + " LM y = " + str(lane_marking_y)
            conn.send(string.encode())
            GPIOBlue()
            return True
    elif(left == 1 and abs(location_y- left_y) <= 1): # if x are negligibly similar
        lane_width = left_lane_width
        lane_marking_y = left_y # constant y
        car_x = location_x
        if(car_x - left_x < 0):
            polarity = -1
            lane_marking_x = left_x - lane_width/2 # different y
        else:
            polarity = 1
            lane_marking_x = left_x + lane_width/2 # different y
        if(polarity == -1 and car_x - lane_marking_x >= -1):
            string = "Warning! Approaching left lane!, car x = " + str(car_x) + " LM x = " + str(lane_marking_x)
            conn.send(string.encode())
            GPIOBlue()
            return True
        elif(polarity == 1 and lane_marking_x - car_x >= -1):
            string = "Warning! Approaching left lane!, car x = " + str(car_x) + " LM x = " + str(lane_marking_x)
            conn.send(string.encode())
            GPIOBlue()
            return True
    return False
def ThreadFunction(conn):
    global d
    while True:
        data = conn.recv(4096)
        d = pickle.loads(data)
def main():
    HOST = '192.168.0.222'
    PORT = 50007
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    print('Connected by', addr)
    thread = threading.Thread(target=ThreadFunction, args=(conn,))
    thread.start()
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup([BTN_B, BTN_R, BTN_G], GPIO.OUT)
    GPIO.output(BTN_B, False)
    GPIO.output(BTN_R, False)
    GPIO.output(BTN_G, False)
    GPIO.setup([LED_B, LED_R, LED_G], GPIO.OUT, initial = GPIO.LOW)
    output = GPIO.HIGH
    GPIOGreen() # for connection successful
    while True:
        try:
            #data = conn.recv(4096)
            #d = pickle.loads(data)
            if(d is not None):
                right_lane_departure_warning(conn, d.location_x, d.location_y, d.right, d.right_x, d.right_y, d.right_lane_width)
                left_lane_departure_warning(conn, d.location_x, d.location_y, d.left, d.left_x, d.left_y, d.left_lane_width)
        except EOFError:
            thread.join()
            s.close()
            exit()
        except KeyboardInterrupt:
            s.close()
            thread.join()
            exit()
        
    conn.close()
main()
