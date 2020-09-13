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

IM_WIDTH = 640 # x
IM_HEIGHT = 480 # y
VEHICLE_X = 320
VEHICLE_Y = 479

i = 1
w = 0
semantic_image_list = []
markings_list = []

def getLaneDistances(location, left_wp, right_wp):
    print(location.x)
    print(location.y)
    print(location.z)
    print(left_wp.transform.location.x)
    print(left_wp.transform.location.y)
    print(left_wp.transform.location.z)
def save_semantic_img(image, name):
    global i
    image.convert(carla.ColorConverter.CityScapesPalette)
    image.save_to_disk(name+"/{}.png".format(i))
    i+=1
def determineLaneDistances():
    for line in roadlines:
        if(line[0]-VEHICLE_X <= 10 and line[1]-VEHICLE_Y <= 10):
            print("Approaching a lane, caution!")

def process_img(image): #live preview
    image.convert(carla.ColorConverter.CityScapesPalette)
    global arr_i3
    arr_i = np.array(image.raw_data)
    arr_i2 = arr_i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    arr_i3 = arr_i2[:, :, :3]
    for i in range(0, 479):
        for j in range(0, 639):
            if(not(arr_i3[i][j][2] == 157 and arr_i3[i][j][1] == 234 and arr_i3[i][j][0] == 50)):
                # image masking
                arr_i3[i][j][0] = 0
                arr_i3[i][j][1] = 0
                arr_i3[i][j][2] = 0
                
    semantic_image_list.append(arr_i3)
    length = len(semantic_image_list)
    #diff(semantic_image_list[length-2], semantic_image_list[length-1])
    cv2.imshow("", arr_i3)
    cv2.waitKey(1)
    return arr_i3/255.0
def diff(image1, image2):
    roadlines_image1 = []
    roadlines_image2 = []
    diff = []
    for i in range(0, 479):
        for j in range(0, 639):
            if((image1[i][j][2] == 157 and image1[i][j][1] == 234 and image1[i][j][0] == 50)):
                roadlines_image1.append([i, j])
    for i in range(0, 479):
        for j in range(0, 639):
            if((image2[i][j][2] == 157 and image2[i][j][1] == 234 and image2[i][j][0] == 50)):
                roadlines_image2.append([i, j])
    l1 = len(roadlines_image1)
    l2 = len(roadlines_image2)
    length = l1 if (l1 < l2) else l2
    for x in (0, length-1):
        diff.append([roadlines_image1[x][0] - roadlines_image2[x][0], roadlines_image1[x][1] - roadlines_image2[x][1]])
    for line in diff:
        if(diff != [0,0]):
            print("Lane crossing")
    #print(diff)
def get_image_difference(image_1, image_2):
    first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
    second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])
    
    img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
    img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
    img_template_diff = 1 - img_template_probability_match

    # taking only 10% of histogram diff, since it's less accurate than template method
    commutative_image_diff = (img_hist_diff / 10) + img_template_diff
    return commutative_image_diff

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
    client = carla.Client('localhost', 2000) # connect to server
    client.set_timeout(500.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('model3')[0] # blueprint for Tesla Model 3
    print(bp)
    spawn_point = random.choice(world.get_map().get_spawn_points()) # choose a random point to spawn the car
    #transform = carla.Transform(carla.Location(x=230, y=155, z=40), carla.Rotation(yaw=-90))
   # print(transform or spawn_point)
    vehicle = world.try_spawn_actor(bp, spawn_point) # spawn the car (actor)
    
    ''' camera
    camera = carla.sensor.Camera('MyCamera', PostProcessing = 'SemanticSegmentation')
    camera.set(FOV=90.0)
    camera.set_image_size(800, 600)
    camera.set_position(x=0.30, y=0, z=1.30)
    camera.set_rotation(pitch=0, yaw=0, roll=0)
    carla_settings.add_sensor(camera)'''
    

    
    # get the blueprint for this sensor
    blueprint = blueprint_library.find('sensor.camera.semantic_segmentation')
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

    #all_waypoints = carla_map.generate_waypoints(distance=1.0)

    #print(location)

   # waypoint = carla_map.get_waypoint(location, project_to_road=True)
    
    actor_list.append(vehicle)
    #vehicle.set_autopilot(True)
    
    #file = open("Waypoints.txt", "w+")
    #logging.basicConfig(filename='newlane.log', filemode='w', level=logging.DEBUG)
    while True:
        m = world.get_map()
        location = vehicle.get_location()
        car = m.get_waypoint(location, project_to_road=True)
        print(str(car))
        print(str(random.choice(car.next(2.0))))
        
        
    ''' while True:
        left = w.get_left_lane()
        right = w.get_right_lane()
        print("Left lane: " + str(left))
        print("Car: " + str(car))
        print("Right lane: " + str(right))
        straight(vehicle, 10, 1.0)
        location = vehicle.get_location()
        car = m.get_waypoint(location, project_to_road=True)
        w = random.choice(car.next(2.0))
        time.sleep(1)'''
    '''for i in range(0, 1000):
        #nearest_waypoint = world.get_map().get_waypoint(spawn_point.location, project_to_road=True)
       # waypoints = nearest_waypoint.next(2.0)
        #file.write("Center of lanes: " + str(waypoints[0]) + "\n")
        #left_wp = nearest_waypoint.get_left_lane()
        #right_wp = nearest_waypoint.get_right_lane()
       # file.write("Left lane: " + str(left_wp) + "\n")
        #file.write("Right lane: " + str(right_wp) + "\n")
        #getLaneDistances(location, left_wp, right_wp)
        time.sleep(1)'''
    time.sleep(500)

   # print(waypoint)
   # ...
    # ...
    # waypoint stuff
    ''' waypoint = world.get_map().get_waypoint(vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
    print("Current lane type: " + str(waypoint.lane_type))
    # Check current lane change allowed
    print("Current Lane change:  " + str(waypoint.lane_change))
    # Left and Right lane markings
    print("L lane marking type: " + str(waypoint.left_lane_marking.type))
    print("L lane marking change: " + str(waypoint.left_lane_marking.lane_change))
    print("R lane marking type: " + str(waypoint.right_lane_marking.type))
    print("R lane marking change: " + str(waypoint.right_lane_marking.lane_change))'''
    # ...
    
    
    
    
    


finally:
    print("Destroying actors ...")
    for actor in actor_list:
        actor.destroy()
    print("Done")
