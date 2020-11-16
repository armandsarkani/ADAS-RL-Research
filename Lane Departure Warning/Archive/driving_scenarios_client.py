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
#from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import carla
import logging
import random
import time
import cv2
import manual_control
import math
import pickle
import socket
import threading


IM_WIDTH = 640 # x
IM_HEIGHT = 480 # y
VEHICLE_X = 320
VEHICLE_Y = 479

i = 1
q = 1
w = 0
semantic_image_list = []
markings_list = []
logger = None
crossed_lane = 0
warning_active = False
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
        if(worldmap.get_waypoint(location).get_right_lane() is not None):
            self.right_x = worldmap.get_waypoint(location).get_right_lane().transform.location.x
            self.right_y = worldmap.get_waypoint(location).get_right_lane().transform.location.y
            self.right_lane_width = worldmap.get_waypoint(location).get_right_lane().lane_width
            self.right = 1
        if(worldmap.get_waypoint(location).get_left_lane() is not None):
            self.left_x = worldmap.get_waypoint(location).get_left_lane().transform.location.x
            self.left_y = worldmap.get_waypoint(location).get_left_lane().transform.location.y
            self.left_lane_width = worldmap.get_waypoint(location).get_left_lane().lane_width
            self.left = 0
def DetermineLaneCentering(prev_location, prev_lane_id):
    global crossed_lane
    if(vehicle is None or worldmap is None):
        return
    location = vehicle.get_location()
    car = worldmap.get_waypoint(location)
    new_lane_id = car.lane_id
    # Vehicle crossed the lane before the response time
    print("Old lane id: ", prev_lane_id)
    print("New lane id: ", new_lane_id)
    if(new_lane_id != prev_lane_id):
        crossed_lane += 1
        print("Vehicle crossed lane! Moving vehicle to center of lane...")
        logger.debug("Vehicle crossed lane " + str(crossed_lane) + " time(s)")
        logger.debug("Moving vehicle to center of lane...\n")
        vehicle.set_transform(car.transform)
        return
    # Vehicle crossed the lane before the response time
    print("OK! Moving vehicle to center of lane...")
    #logger.debug("Vehicle was at: x = " + str(prev_location.x) + "y = " + str(prev_location.y))
    #logger.debug("Center of lane is at (wp): x = " +  str(car.transform.location.x) + "y = " + str(car.transform.location.y))
    logger.debug("Vehicle did not cross lane")
    logger.debug("Moving vehicle to center of lane...\n")
    vehicle.set_transform(car.transform)
def Receive(sock):
    time_issued_warning = 0
    global warning_active
    while(True):
        prev_location = None
        prev_lane_id = None
        if(vehicle is not None and worldmap is not None):
            prev_lane_id = worldmap.get_waypoint(vehicle.get_location()).lane_id
            prev_location = vehicle.get_location()
        response = sock.recv(4096)
        if("RightSafe" not in response.decode()):
            print("Warning issued")
            time_issued_warning += 1
            logger.debug(response.decode())
            logger.debug("Brake: " + str(vehicle.get_control().brake))
            logger.debug("Steering: " + str(vehicle.get_control().steer))
            logger.debug("Throttle: " + str(vehicle.get_control().throttle))
            logger.debug("Warning issued " + str(time_issued_warning) + " time(s)\n")
            warning_active = True
            while(warning_active): # waiting during response time
                pass
            #DetermineLaneCentering(prev_location, prev_lane_id)
def getLaneDistances(location, left_wp, right_wp):
    print(location.x)
    print(location.y)
    print(location.z)
    print(left_wp.transform.location.x)
    print(left_wp.transform.location.y)
    print(left_wp.transform.location.z)
def save_semantic_img(image, name):
    global i
    #image.convert(carla.ColorConverter.CityScapesPalette)
    image.save_to_disk(name+"/{}.png".format(i))
    i+=1

def process_img(image): #live preview
    #image.convert(carla.ColorConverter.CityScapesPalette)
    global arr_i3
    arr_i = np.array(image.raw_data)
    arr_i2 = arr_i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    arr_i3 = arr_i2[:, :, :3]
    cv2.imshow("", arr_i3)
    cv2.waitKey(1)
    d = LaneDepartureData()
    data_string = pickle.dumps(d)
    sock.send(data_string)
    return arr_i3/255.0
def straight(vehicle, t, throttle):
    print("Going straight ...")
    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=0)) # car only going forward
    time.sleep(t)
def left(vehicle, t, throttle, steer):
    print("Turning left ...")
    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=-steer))
    time.sleep(t)
def right(vehicle, t, throttle, steer):
    print("Turning right ...")
    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
    time.sleep(t)

actor_list = []
try:
    if(sys.argv[1] == "help"):
       print("Format: (device) (throttle) (set/none) (type of driver)")
       exit()
    device = sys.argv[1]
    throttle = float(sys.argv[2])
    worldset = sys.argv[3]
    driver = sys.argv[4]
    if(device == "iMac"):
        HOST = '192.168.0.4' # iMac Pro
    elif(device == "MBPo"):
        HOST = '192.168.254.41' # 16-inch other
    elif(device == "MBP"):
        HOST = '192.168.0.78' # 16-inch
    else:
        HOST = 'localhost'
    PORT = 50007
    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    thread = threading.Thread(target=Receive, args=(sock,))
    client = carla.Client('localhost', 2000) # connect to server
    client.set_timeout(20)
    global worldmap
    if(worldset == "set"): # first time, load Town06
        world = client.load_world('Town06') # highway town simple
    else: # otherwise, continue using Town06
        world = client.get_world()
    worldmap = world.get_map()
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('model3')[0] # blueprint for Tesla Model 3
    bp.set_attribute('color', '0,0,255') # blue
    bp2 = blueprint_library.filter('prius')[0] # blueprint for red Toyota Prius
    bp3 = blueprint_library.filter('mustang')[0] # blueprint for light blue Ford Mustang
    print(bp)
    global vehicle
    vehicle = None
    spawn_point = carla.Transform(carla.Location(151.071,147.458,2.5),carla.Rotation(0,0.234757,0))
    spawn_point_2 = carla.Transform(carla.Location(124.071,150.58,2.5),carla.Rotation(0,0.234757,0))
    spawn_point_3 = carla.Transform(carla.Location(154.071,150.458,2.5),carla.Rotation(0,0.234757,0))
    #spawn_point = random.choice(world.get_map().get_spawn_points())
    print(str(spawn_point.location) + str(spawn_point.rotation))
    vehicle = world.try_spawn_actor(bp, spawn_point) # spawn the car (actor)
    vehicle_2 = world.try_spawn_actor(bp2, spawn_point_2) # spawn the car (actor)
    vehicle_3 = world.try_spawn_actor(bp3, spawn_point_3) # spawn the car (actor)
    # get the blueprint for this sensor
    blueprint = blueprint_library.find('sensor.camera.rgb')
    #blueprint.set_attribute('post_processing', 'SemanticSegmentation')
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint.set_attribute('fov', '110')
    

    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)

    # do something with this sensor
    #sensor.listen(lambda data: save_semantic_img(data, "semantic"))
    sensor.listen(lambda data: process_img(data))
    
    # lane detection
    # 1. Get the carla.Map object from carla.World using
    carla_map = world.get_map()

    actor_list.append(vehicle)
    actor_list.append(vehicle_2)
    actor_list.append(vehicle_3)

    #file = open("Waypoints.txt", "w+")
    logging.basicConfig(filename="DrivingScenarios.log", format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    thread.start()
    while(True):
        location = vehicle.get_location()
        car = worldmap.get_waypoint(location) # center of lane waypoint
        if(warning_active and driver == "slow"):
            init_time = time.time()
            response = sock.recv(4096)
            while("RightSafe" not in response.decode()):
                print("Warning active and slow!")
                logger.debug("Warning active and slow!")
                vehicle.apply_control(carla.VehicleControl(throttle=random.uniform(throttle*0.25, throttle*0.75), steer=random.uniform(-0.03, -0.05)))
                vehicle_2.apply_control(carla.VehicleControl(throttle=throttle, steer=0))
                vehicle_3.apply_control(carla.VehicleControl(throttle=throttle, steer=0))
                response = sock.recv(4096)
            print("Response time elapsed (s): ", time.time() - init_time)
            logger.debug("Response time elapsed (s): " + str(time.time() - init_time) + "\n")
            location = vehicle.get_location()
            car = worldmap.get_waypoint(location)
            vehicle.set_transform(car.transform) # quick adjust wheels
            warning_active = False
            print("Warning off!")
        if(warning_active and driver == "cautious"):
            init_time = time.time()
            response = sock.recv(4096)
            straight_time = time.time()
            total_time = random.uniform(0.5, 3.5)
            while(time.time() - straight_time < total_time):
                vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=0))
            while("RightSafe" not in response.decode()):
                print("Warning active and cautious!")
                logger.debug("Warning active and cautious!")
                vehicle.apply_control(carla.VehicleControl(throttle=random.uniform(throttle*0.25, throttle*0.75), steer=random.uniform(-0.03,-0.05)))
                vehicle_2.apply_control(carla.VehicleControl(throttle=throttle, steer=0))
                vehicle_3.apply_control(carla.VehicleControl(throttle=throttle, steer=0))
                response = sock.recv(4096)
            print("Response time elapsed (s): ", time.time() - init_time)
            logger.debug("Response time elapsed (s): " + str(time.time() - init_time) + "\n")
            location = vehicle.get_location()
            car = worldmap.get_waypoint(location)
            vehicle.set_transform(car.transform) # quick adjust wheels
            warning_active = False
            print("Warning off!")
        if(warning_active and driver == "fast"):
            init_time = time.time()
            response = sock.recv(4096)
            while("RightSafe" not in response.decode()):
                print("Warning active and fast!")
                logger.debug("Warning active and fast!")
                vehicle.apply_control(carla.VehicleControl(throttle=random.uniform(throttle*1.25,throttle*2.5), steer=random.uniform(-0.1,-0.2)))
                vehicle_2.apply_control(carla.VehicleControl(throttle=throttle, steer=0))
                vehicle_3.apply_control(carla.VehicleControl(throttle=throttle, steer=0))
                response = sock.recv(4096)
            print("Response time elapsed (s): ", time.time() - init_time)
            logger.debug("Response time elapsed (s): " + str(time.time() - init_time) + "\n")
            location = vehicle.get_location()
            car = worldmap.get_waypoint(location)
            vehicle.set_transform(car.transform) # quick adjust wheels
            warning_active = False
            print("Warning off!")
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=0.001))
        vehicle_2.apply_control(carla.VehicleControl(throttle=throttle, steer=0))
        vehicle_3.apply_control(carla.VehicleControl(throttle=throttle, steer=0))
        if(vehicle.is_at_traffic_light()):
            print("At traffic light...")
        while(vehicle.is_at_traffic_light()):
            vehicle.apply_control(carla.VehicleControl(brake=1.0))
    
except KeyboardInterrupt:
    sensor.stop()
    print("Destroying actors ...")
    for actor in actor_list:
        actor.destroy()
    print("Done")
    thread.join()
    cv2.destroyAllWindows()


