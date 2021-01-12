from __future__ import print_function
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

import matplotlib.pyplot as plt
import numpy as np
import carla
import random
import time
import math
import pickle
import socket
import threading
import argparse
from datetime import datetime, date
import raycast_sensor_testing as raycast

# miscellaneous
cautious_time = 1.5
stop_thread = False
corrective_percentage = 0.95
turn_signal_status = False
locationx = 0
locationy = 0
locationz = 0
collisions = []

# lane departure data class to send to server
class LaneDepartureData:
    def __init__(self):
        location = vehicle.get_location()
        self.location_x = vehicle.get_location().x
        self.location_y = vehicle.get_location().y
        self.right = 0
        self.left = 0
        self.right_x = 0
        self.right_y = 0
        self.right_lane_width = 0
        self.left_x = 0
        self.left_y = 0
        self.acc_x = vehicle.get_acceleration().x
        self.acc_y = vehicle.get_acceleration().y
        self.acc_z  = vehicle.get_acceleration().z
        self.left_lane_width = 0
        self.steer = vehicle.get_control().steer
        velocity = vehicle.get_velocity()
        self.vel_x = velocity.x
        self.vel_y = velocity.y
        self.vel_z = velocity.z
        self.lane_id = worldmap.get_waypoint(location).lane_id
        self.speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        self.speed_limit = vehicle.get_speed_limit()
        self.turn_signal = turn_signal_status
        if(worldmap.get_waypoint(location).get_right_lane() is not None):
            self.right_x = worldmap.get_waypoint(location).get_right_lane().transform.location.x
            self.right_y = worldmap.get_waypoint(location).get_right_lane().transform.location.y
            self.right_lane_width = worldmap.get_waypoint(location).get_right_lane().lane_width
            self.right = 1
        if(worldmap.get_waypoint(location).get_left_lane() is not None):
            self.left_x = worldmap.get_waypoint(location).get_left_lane().transform.location.x
            self.left_y = worldmap.get_waypoint(location).get_left_lane().transform.location.y
            self.left_lane_width = worldmap.get_waypoint(location).get_left_lane().lane_width
            self.left = 1

# helpers from server
def right_lane_distance(location_x, location_y, right_x, right_y, right_lane_width):
    if(abs(location_x - right_x) <= 1): # if x are negligibly similar
        if(location_y - right_y < 0):
            lane_marking_y = right_y - right_lane_width/2
            return(location_y - lane_marking_y)
        else:
            lane_marking_y = right_y + right_lane_width/2
            return(lane_marking_y - location_y)
    elif(abs(location_y - right_y) <= 1): # if y are negligibly similar
        if(location_x - right_x < 0):
            lane_marking_x = right_x - right_lane_width/2
            return(location_x - lane_marking_x)
        else:
            lane_marking_x = right_x + right_lane_width/2
            return(lane_marking_x - location_x)
def left_lane_distance(location_x, location_y, left_x, left_y, left_lane_width):
    if(abs(location_x - left_x) <= 1): # if x are negligibly similar
           if(location_y - left_y < 0):
               lane_marking_y = left_y - left_lane_width/2
               return(location_y - lane_marking_y)
           else:
               lane_marking_y = left_y + left_lane_width/2
               return(lane_marking_y - location_y)
    elif(abs(location_y - left_y) <= 1): # if y are negligibly similar
           if(location_x - left_x < 0):
               lane_marking_x = left_x - left_lane_width/2
               return(location_x - lane_marking_x)
           else:
               lane_marking_x = left_x + left_lane_width/2
               return(lane_marking_x - location_x)

            
# socket communication functions
def send(sock):
    time_issued_warning = 0
    while(True):
        send_data()
        if(stop_thread):
            break
def send_data(): # send data to server
    d = LaneDepartureData()
    data_string = pickle.dumps(d)
    sock.send(data_string)
    
# driver scenarios
def slow_driver(throttle, behavior):
    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=behavior*random.uniform(-0.005,-0.015)))
def cautious_driver(throttle, behavior):
    init_time = time.time()
    while(time.time() - init_time < cautious_time):
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=0))
    for i in range(0, 3):
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=behavior*random.uniform(-0.005,-0.015)))
def fast_driver(throttle, behavior):
    vehicle.apply_control(carla.VehicleControl(throttle=throttle*1.1, steer=behavior*random.uniform(-0.01,-0.025)))
def drowsy_driver(throttle, behavior):
    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=behavior*random.uniform(-0.015,0.015)))

# helper functions
def quick_lane_centering(vehicle):
    location = vehicle.get_location()
    lane_center = worldmap.get_waypoint(location)
    vehicle.set_transform(lane_center.transform)
def parse_arguments():
    argparser = argparse.ArgumentParser(
            description='Q-learning LDW Server')
    argparser.add_argument(
        '-n', '--hostname',
        metavar='NAME',
        default='localhost',
        help='computer hostname or IP address')
    argparser.add_argument(
        '-m', '--mode',
        metavar='MODE',
        default='continue',
        help='world mode (set/continue)')
    argparser.add_argument(
        '-t', '--throttle',
        metavar='#.#',
        default=0.5,
        help='constant throttle value for vehicle to drive at (range: 0.0 - 1.0)')
    argparser.add_argument(
        '-d', '--driver',
        metavar='DRIVER',
        default='fast',
        help='type of driver response time (fast, slow, cautious)')
    argparser.add_argument(
        '-x', '--locationx',
        metavar='LOCATIONX',
        default=151.071,
        help='x-coordinate of spawn location')
    argparser.add_argument(
        '-y', '--locationy',
        metavar='LOCATIONY',
        default=143.458,
        help='y-coordinate of spawn location')
    argparser.add_argument(
        '-z', '--locationz',
        metavar='LOCATIONZ',
        default=2.5,
        help='z-coordinate of spawn location')
    argparser.add_argument(
        '-a', '--autonomous',
        metavar='AUTO',
        default='off',
        help='turn on/off autonomous driving')
    argparser.add_argument(
        '-dn', '--name',
        metavar='DRIVER_NAME',
        default='DefaultDriver',
        help='driver name')
    argparser.add_argument(
        '-v', '--vehicle',
        metavar='VEHICLE',
        default='model3',
        help='vehicle blueprint')
    argparser.add_argument(
        '-b', '--behavior',
        metavar='BEHAVIOR',
        default='right',
        help='vehicle polarity behavior (approaching left or right)')
    args = argparser.parse_args()
    return args

# process sensory data
def process_lidar(measurement):
    for detection in measurement:
        print(detection)
def collision_handler(event):
    global collisions
    collisions.append(event)
    print("Collision")
    colliding_actor = event.other_actor
    impulse = event.normal_impulse
    intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)

# game loop
def script_loop(driver, throttle, threshold, behavior):
    global turn_signal_status
    while(True):
        response = sock.recv(4096)
        print(response.decode())
        flag = False
        d = LaneDepartureData()
        dr = right_lane_distance(d.location_x, d.location_y, d.right_x, d.right_y, d.right_lane_width)
        while(dr is None):
            d = LaneDepartureData()
            dr = right_lane_distance(d.location_x, d.location_y, d.right_x, d.right_y, d.right_lane_width)
        dr *= -1
        if(dr <= threshold and np.random.binomial(1, 0) == 1): # turn on turn signals
            turn_signal_status = True
            print("Turn signals on ...")
            orig_lane_id = worldmap.get_waypoint(vehicle.get_location()).lane_id
            while(worldmap.get_waypoint(vehicle.get_location()).lane_id == orig_lane_id):
                vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=0.05))
            quick_lane_centering(vehicle)
            print("Turn signals off.")
            turn_signal_status = False
            continue
        secondary_corrective_percentage = 0.85
        while(np.random.binomial(1, corrective_percentage) == 1 and "WARNING! Approaching lane." in response.decode()): # only take corrective action certain % of time
            if(dr >= threshold and np.random.binomial(1, secondary_corrective_percentage) == 1):  # certain % of the time, if driver is at threshold or closer to the center of the lane, do not take a corrective action
                print("Not doing corrective action.")
                break # do not take a corrective action
            if(driver == "slow"):
                slow_driver(throttle, behavior)
            if(driver == "cautious"):
                cautious_driver(throttle, behavior)
            if(driver == "drowsy"):
                drowsy_driver(throttle, behavior)
            if(driver == "fast"):
                fast_driver(throttle, behavior)
            flag = True
            response = sock.recv(4096)
            print(response.decode())
        if(flag):
           quick_lane_centering(vehicle)
        if(vehicle.get_location().x >= 630.0):
            vehicle.set_transform(carla.Transform(carla.Location(locationx,locationy,locationz),carla.Rotation(0,0.234757,0))) # reset position to the beginning of the road to continue testing when the vehicle reaches end of road
        if(driver == "drowsy"):
            throttle *= 0.95
            if(throttle < orig_throttle*0.5):
                throttle = orig_throttle
            steer = random.uniform(-0.0005, 0.001)
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
        else:
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=behavior*0.001))
    
# main function
def main():
    actor_list = []
    global stop_thread
    global locationx, locationy, locationz
    try:
        args = parse_arguments()
        locationx = float(args.locationx)
        locationy = float(args.locationy)
        locationz = float(args.locationz)
        behavior = 1 if (args.behavior == 'right') else -1
        hostname_to_IP = {'iMac': '192.168.0.2', 'MBP': '192.168.0.78', 'MBPo': '192.168.254.41', 'MBAo': '192.168.254.67', 'localhost': '127.0.0.1'}
        IP = hostname_to_IP.get(args.hostname)
        if(IP is None):
             IP = args.hostname
        worldset = args.mode
        driver_name = args.name
        throttle, orig_throttle = float(args.throttle), float(args.throttle)
        driver = args.driver
        port = 50007
        global sock
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((IP, port))
        if(driver_name != "DefaultDriver"):
            sock.send(driver_name.encode())
            response = sock.recv(4096)
            while(response is None or "Success" not in response.decode()):
                response = sock.recv(4096)
                if(response is not None and "Success" in response.decode()):
                    break
        thread = threading.Thread(target=send, args=(sock,))
        client = carla.Client('localhost', 2000) # connect to server
        client.set_timeout(20)
        global worldmap
        if(worldset == 'set'): # first time, load Town06
            world = client.load_world('Town06') # highway town simple
        else: # otherwise, continue using Town06
            world = client.get_world()
        worldmap = world.get_map()
        if(args.vehicle == 'random'):
            bp = random.choice(world.get_blueprint_library().filter('vehicle.*')) # blueprint for random vehicle
        else:
            bp = world.get_blueprint_library().filter(args.vehicle)[0] # blueprint for other vehicle
        global vehicle
        vehicle = None
        spawn_point = carla.Transform(carla.Location(locationx,locationy,locationz),carla.Rotation(0,0.234757,0))
        vehicle = world.try_spawn_actor(bp, spawn_point) # spawn the car (actor)
        collision_sensor = world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'),
                                        carla.Transform(), attach_to=vehicle)
        collision_sensor.listen(lambda event: collision_handler(event))
        actor_list.append(vehicle)
        if(args.autonomous == 'on'):
            vehicle.set_autopilot(True)
            thread.start()
            while True:
                response = sock.recv(4096)
                if(response is not None):
                    print(response.decode())
        thread.start()
        time.sleep(3)
        threshold_dict = {"drowsy": 1.3, "slow": 1.2, "cautious": 1.1, "fast": 0.65}
        threshold = threshold_dict.get(driver)
        script_loop(driver, throttle, threshold, behavior)
    except KeyboardInterrupt:
        if(len(collisions) > 0):
            print("Run completed.", len(collisions), "collisions measured.")
        else:
            print("No collisions measured.")
        for actor in actor_list:
            actor.destroy()
        print("Actors destroyed.")
        stop_thread = True
        exit()
main()



