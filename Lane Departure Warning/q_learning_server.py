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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import trange

# probability for exploration
epsilon = 0.1
# step size
alpha = 0.5
# gamma for Q-Learning
gamma = 0.99
# rate at which state transitions are measured
sampling_rate = 0.1
# binary actions
actions = [0, 1] # 0 = no warning, 1 = issue warning
# number of states per dimension
num_distance_states = 12
num_speed_states = 3
# q-value lookup table, initialized to zeros
q_values = np.zeros((num_distance_states, num_speed_states, 2))
# total rewards
rewards = 0
# iterations for runtime
iterations = 100
# connection variables
conn = None
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
    speed = 0
    speed_limit = 0
def receive_metrics():
    while True:
        if(d is not None):
            dr = right_lane_distance(d.location_x, d.location_y, d.right_x, d.right_y, d.right_lane_width)
            dl = left_lane_distance(d.location_x, d.location_y, d.left_x, d.left_y, d.left_lane_width)
            if(dr is not None and dl is not None and d.speed_limit is not None and d.speed is not None):
                dr *= -1
                dl *= -1
                dr = float('%.2f' % dr)
                dl = float('%.2f' % dl)
                speed = d.speed * 2.237 # m/s to mph
                return dl, dr, speed, d.speed_limit
def initialize_q_table():
    for i in range(0, 5):
        q_values[i, :, 0] = 0.5
        q_values[i, :, 1] = 0
    for i in range(5, 8):
        q_values[i, :, 0] = 0.5
        q_values[i, :, 1] = 0.25
    for i in range(8, 12):
        q_values[i, :, 0] = 0
        q_values[i, :, 1] = 0.5
def enumerate_state(dl, dr, speed, speed_limit):
    # initialize state vector
    state = [0, 0]
    # distance
    if(dl == 1.75 and dr == 1.75):
        state[0] = 0
    elif((dl >= 1.6 and dl < 1.75) or (dr >= 1.6 and dr < 1.75)):
        state[0] = 1
    elif((dl >= 1.45 and dl < 1.6) or (dr >= 1.45 and dr < 1.6)):
        state[0] = 2
    elif((dl >= 1.3 and dl < 1.45) or (dr >= 1.3 and dr < 1.45)):
        state[0] = 3
    elif((dl >= 1.15 and dl < 1.3) or (dr >= 1.15 and dr < 1.3)):
        state[0] = 4
    elif((dl >= 1.0 and dl < 1.15) or (dr >= 1.0 and dr < 1.15)):
        state[0] = 5
    elif((dl >= 0.85 and dl < 1.0) or (dr >= 0.85 and dr < 1.0)):
        state[0] = 6
    elif((dl >= 0.7 and dl < 0.85) or (dr >= 0.7 and dr < 0.85)):
        state[0] = 7
    elif((dl >= 0.55 and dl < 0.7) or (dr >= 0.55 and dr < 0.7)):
        state[0] = 8
    elif((dl >= 0.4 and dl < 0.55) or (dr >= 0.4 and dr < 0.55)):
        state[0] = 9
    elif((dl >= 0.25 and dl < 0.4) or (dr >= 0.25 and dr < 0.4)):
        state[0] = 10
    elif(dl < 0.25 or dr < 0.25):
        state[0] = 11
    # speed
    if(speed <= speed_limit * 0.9):
        state[1] = 0
    elif(speed > 0.9 * speed_limit and speed <= 1.1 * speed_limit):
        state[1] = 1
    elif(speed > 1.1 * speed_limit):
        state[1] = 2
    return state
def define_rewards(state, next_state):
    reward = 0
    if(state[0] > next_state[0]):
        reward += 10 * (state[0] - next_state[0])
    elif(state[0] < next_state[0]):
        reward += -10 * (next_state[0] - state[0])
    if(state[1] > next_state[1]):
        reward += 15 * (state[1] - next_state[1])
    elif(state[1] < next_state[1]):
        reward += -15 * (next_state[1] - state[1])
    return reward
def choose_action(state):
    if(np.random.binomial(1, epsilon) == 1):
        return np.random.choice(actions)
    else:
        values = q_values[state[0], state[1], :] # row of values for a given state, any actions
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])
def send_action(action):
    if(action == 1):
        string = "WARNING! Approaching lane."
        conn.send(string.encode())
    else:
        string = "Safe"
        conn.send(string.encode())
# poll action vector every 20 seconds
def q_learning(step_size= alpha):
    global q_values
    global rewards
    rewards = 0
    iteration_rewards = 0
    vector = []
    dl, dr, speed, speed_limit = receive_metrics()
    init_state = enumerate_state(dl, dr, speed, speed_limit)
    for i in trange(iterations):
        state_vector = []
        action = choose_action(init_state)
        send_action(action)
        for j in range(0, int(2/sampling_rate)):
            dl, dr, speed, speed_limit = receive_metrics()
            state = enumerate_state(dl, dr, speed, speed_limit)
            state_vector.append(state)
            time.sleep(sampling_rate)
            dl, dr, speed, speed_limit = receive_metrics()
            next_state = enumerate_state(dl, dr, speed, speed_limit)
            state_vector.append(next_state)
        #iteration_rewards += define_rewards(init_state, state_vector[0])
        #for k in range(0, len(state_vector) - 1):
            #iteration_rewards += define_rewards(state_vector[k], state_vector[k+1])
        #rewards += iteration_rewards
        # Q-learning lookup table update
        iteration_rewards = define_rewards(init_state, state_vector[-1])
        final_state = state_vector[-1]
        print("Going from state", init_state, "to state", final_state, "rewards = ", iteration_rewards)
        delta = step_size * (iteration_rewards + gamma * np.max(q_values[final_state[0], final_state[1], :]) - q_values[init_state[0], init_state[1], action])
        q_values[init_state[0], init_state[1], action] += delta
        print("Delta =", delta)
        np.save("DriverQValues.npy", q_values) # save on each iteration
        init_state = final_state
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
def ThreadFunction(conn):
    global d
    global conn_reset
    while True:
        try:
            data = conn.recv(4096)
            d = pickle.loads(data)
        except ConnectionResetError:
            conn_reset = True
            np.save("DriverQValues.npy", q_values) # custom file
            print("Disconnected.")
            break
        except EOFError:
            conn_reset = True
            np.save("DriverQValues.npy", q_values) # custom file
            print("Disconnected.")
            break
        except BrokenPipeError:
            conn_reset = True
            np.save("DriverQValues.npy", q_values) # custom file
            print("Disconnected.")
            break
def main():
    global conn_reset
    global conn
    global q_values
    global rewards
    device = sys.argv[1]
    if(device == "iMac"):
        HOST = '192.168.0.5' # iMac Pro
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
    if(os.path.exists("DriverQValues.npy")):
        print("\n")
        print("Existing Q-table loaded ...")
        q_values = np.load("DriverQValues.npy")
        np.set_printoptions(suppress=True)
        print(q_values)
    else:
        np.save("DriverQValues.npy", q_values) # custom file
        #initialize_q_table()
    try:
        episode = 1
        while(True):
            print("Running episode " + str(episode) + " (" + str(iterations) + " iterations)")
            q_learning()
            print("Episode " + str(episode) + " completed. Total rewards this episode = ", rewards)
            episode += 1
            print("\n")
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
