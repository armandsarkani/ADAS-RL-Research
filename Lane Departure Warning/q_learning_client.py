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


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
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

# warning status
stop_thread = False
corrective_percentage = 0.5
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
        self.lane_id = worldmap.get_waypoint(location).lane_id
        self.speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        self.speed_limit = vehicle.get_speed_limit()
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
            
# socket communication functions
def Receive(sock):
    time_issued_warning = 0
    while(True):
        response = sock.recv(4096)
        if(stop_thread):
            break
def send_data(): # send data to server
    d = LaneDepartureData()
    data_string = pickle.dumps(d)
    sock.send(data_string)
    
# driver scenarios
def slow_driver(throttle):
    init_time = time.time()
    response = sock.recv(4096)
    while("Safe" not in response.decode()):
        print("Warning active and slow!")
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=random.uniform(-0.001, -0.005)))
        send_data()
        response = sock.recv(4096)
    print("Response time elapsed (s): ", time.time() - init_time)
    location = vehicle.get_location()
    lane_center = worldmap.get_waypoint(location)
    vehicle.set_transform(lane_center.transform) # quick adjust wheels
    print("Warning off!")
def cautious_driver(throttle):
    init_time = time.time()
    response = sock.recv(4096)
    straight_time = time.time()
    total_time = random.uniform(0.5, 1.5)
    while(time.time() - straight_time < total_time):
        send_data()
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=0))
    while("Safe" not in response.decode()):
        print("Warning active and cautious!")
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=random.uniform(-0.001,-0.003)))
        send_data()
        response = sock.recv(4096)
    print("Response time elapsed (s): ", time.time() - init_time)
    location = vehicle.get_location()
    lane_center = worldmap.get_waypoint(location)
    vehicle.set_transform(lane_center.transform) # quick adjust wheels
    print("Warning off!")
def fast_driver(throttle):
    init_time = time.time()
    response = sock.recv(4096)
    while("Safe" not in response.decode()):
        print("Warning active and fast!")
        vehicle.apply_control(carla.VehicleControl(throttle=throttle*1.1, steer=random.uniform(-0.002,-0.005)))
        send_data()
        response = sock.recv(4096)
    print("Response time elapsed (s): ", time.time() - init_time)
    location = vehicle.get_location()
    lane_center = worldmap.get_waypoint(location)
    vehicle.set_transform(lane_center.transform) # quick adjust wheels
    print("Warning off!")

# main function
def main():
    actor_list = []
    global stop_thread
    try:
        argparser = argparse.ArgumentParser(
            description='Q-learning LDW Server')
        argparser.add_argument(
            '-n', '--hostname',
            metavar='NAME',
            default='localhost',
            help='computer short hostname')
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
        args = argparser.parse_args()
        hostname_to_IP = {'iMac': '192.168.0.5', 'MBP': '192.168.0.78', 'MBPo': '192.168.254.41', 'localhost': '127.0.0.1'}
        IP = hostname_to_IP.get(args.hostname)
        worldset = args.mode
        throttle = float(args.throttle)
        driver = args.driver
        port = 50007
        global sock
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((IP, port))
        thread = threading.Thread(target=Receive, args=(sock,))
        client = carla.Client('localhost', 2000) # connect to server
        client.set_timeout(20)
        global worldmap
        if(worldset == 'set'): # first time, load Town06
            world = client.load_world('Town06') # highway town simple
        else: # otherwise, continue using Town06
            world = client.get_world()
        worldmap = world.get_map()
        bp = world.get_blueprint_library().filter('model3')[0] # blueprint for Tesla Model 3
        global vehicle
        vehicle = None
        spawn_point = carla.Transform(carla.Location(151.071,147.458,2.5),carla.Rotation(0,0.234757,0))
        vehicle = world.try_spawn_actor(bp, spawn_point) # spawn the car (actor)
        actor_list.append(vehicle)
        thread.start()
        time.sleep(3)
        while(True):
            send_data()
            response = sock.recv(4096)
            if(np.random.binomial(1, corrective_percentage) == 1 and "Safe" not in response.decode()): # only take corrective action certain % of time
                if(driver == "slow"):
                    slow_driver(throttle)
                if(driver == "cautious"):
                    cautious_driver(throttle)
                if(driver == "fast"):
                    fast_driver(throttle)
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=0.001))
    except KeyboardInterrupt:
        for actor in actor_list:
            actor.destroy()
        print("Actors destroyed.")
        stop_thread = True
        #sock.close()
        exit()
main()



