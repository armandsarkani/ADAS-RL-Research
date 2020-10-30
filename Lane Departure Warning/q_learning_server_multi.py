import glob
import os
import sys
import client_class
import socket
import pickle
import math
import time
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import trange
import argparse
import json
import logging
import statistics
import uuid
from datetime import datetime, date

# thread lock
lock = None

# step size
alpha = 0.6

# gamma for Q-Learning
gamma = 0.8

# rate at which state transitions are measured
sampling_rate = 0.1
vector_size = int(2/sampling_rate)

# human states
attentive = 0
moderate = 1
inattentive = 2
dict_human_states = {"attentive": attentive, "moderate": moderate, "inattentive": inattentive}

# iterations per episode
iterations = 100

# actions
no_warning = 0
warning = 1
actions = [no_warning, warning]
dict_actions = {no_warning: "no warning", warning: "warning"}


# class definitions
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
    lane_id = 0
    turn_signal = None
class State:
    def __init__(self, client):
        self.metrics = calculate_metrics(client)
        self.value = enumerate_state(client, self.metrics)
        
# Q-learning functions
def calculate_metrics(client):
    while True:
        d = client.ldw_data
        if(d is not None):
            while(d is not None and d.turn_signal == True):
                pass
            dr = right_lane_distance(d.location_x, d.location_y, d.right_x, d.right_y, d.right_lane_width)
            dl = left_lane_distance(d.location_x, d.location_y, d.left_x, d.left_y, d.left_lane_width)
            if(dr is not None and dl is not None and d.speed_limit is not None and d.speed is not None and d.lane_id is not None and d.steer is not None):
                dr *= -1
                dl *= -1
                dr = float('%.2f' % dr)
                dl = float('%.2f' % dl)
                speed = d.speed * 2.237 # m/s to mph
                return {"dl": dl, "dr": dr, "speed": speed, "speed_limit": d.speed_limit, "lane_id": d.lane_id, "steer": d.steer}
def enumerate_state(client, metrics):
    # initialize state vector
    state = [0, 0, 0]
    # get values from dictionary
    dl = metrics.get("dl")
    dr = metrics.get("dr")
    speed = metrics.get("speed")
    speed_limit = metrics.get("speed_limit")
    steer = metrics.get("steer")
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
    # human
    state[2] = dict_human_states.get(receive_human_state(client, client.input_file))
    return state
def define_rewards(client, state, action, next_state):
    reward = 0
    if(state.value[0] > next_state.value[0]):
        reward += 10 * (state.value[0] - next_state.value[0])
    elif(state.value[0] < next_state.value[0]):
        reward += -10 * (next_state.value[0] - state.value[0])
    if(state.value[1] > next_state.value[1]):
        reward += 15 * (state.value[1] - next_state.value[1])
    elif(state.value[1] < next_state.value[1]):
        reward += -15 * (next_state.value[1] - state.value[1])
    lane_id = state.metrics.get("lane_id")
    next_lane_id = next_state.metrics.get("lane_id")
    if(lane_id != next_lane_id): # if lane invasion occurs
        reward -= 50
        client.num_invasions += 1
        return reward
    if(is_intermediate(state.value) and is_intermediate(next_state.value) and state.value[0] <= next_state.value[0] and action == warning): # if warning "ignored"
        reward -= 20
    if(is_unsafe(state.value) and is_safe(next_state.value)): # if corrective action taken after warning from unsafe to safe state (warning implied)
        reward += 50
    if(is_unsafe(state.value) and is_intermediate(next_state.value)): # if corrective action taken after warning from unsafe to intermediate state (warning implied)
        reward += 30
    if(is_intermediate(state.value) and is_safe(next_state.value) and action == no_warning): # if corrective action taken after no warning from intermediate to safe state
        reward -= 10
    if(is_intermediate(state.value) and is_unsafe(next_state.value) and action == no_warning): # if no warning issued, but next state was unsafe
        reward -= 30
        if(state.value[2] == inattentive):
            reward -= 10
    if(is_intermediate(state.value) and is_unsafe(next_state.value) and action == warning): # if warning issued, but next state was unsafe
        reward += 20
        if(state.value[2] == inattentive):
            reward += 20
    if(state.value[2] == inattentive and action == warning): # giving warning to inattentive driver
        reward += 5
    if(state.value[2] == attentive and action == warning): # giving warning to attentive driver
        reward -= 5
    if(state.value[2] == attentive and action == no_warning): # not giving warning to attentive driver
        reward += 5
    if(state.value[2] == inattentive and state.metrics.get("steer") > 0.0005 and action == no_warning):
        reward -= 20
    return reward
def is_safe(state_value):
    if(state_value[0] < 3):
        return True
    else:
        return False
def is_intermediate(state_value):
    if(not is_safe(state_value) and not is_unsafe(state_value)):
        return True
    else:
        return False
def is_unsafe(state_value):
    if(state_value[0] > 8 and state_value[2] != inattentive):
        return True
    elif(state_value[0] >= 7 and state_value[2] == inattentive):
        return True
    else:
        return False
def choose_action(client, state_value):
    if(np.random.binomial(1, client.epsilon) == 1):
        return np.random.choice(actions)
    else:
        values = client.q_values[state_value[0], state_value[1], state_value[2], :] # row of values for a given state, any actions
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])
def send_action(conn, action, state):
    if(action == warning):
        string = "WARNING! Approaching lane. State: " + str(state.value)
        conn.send(string.encode())
    else:
        string = "Safe. State: " + str(state.value)
        conn.send(string.encode())
def q_learning(client, conn, thread, episode, step_size= alpha):
    iteration_rewards = 0
    init_state = State(client)
    plot_data = {}
    desc_str = client.driver_name + " (episode " + str(episode) + ")"
    for i in trange(iterations, desc= desc_str, leave=True):
        state_vector = []
        if(client.control and init_state.value[0] == 11): # only when invaded
            action = warning
        elif(client.control and init_state.value[0] != 11):
            action = no_warning
        elif(is_safe(init_state.value)):
            action = no_warning
        elif(is_unsafe(init_state.value)):
            action = warning
        else:
            action = choose_action(client, init_state.value)
        if(action == warning):
            client.warning_states.append(init_state)
        plot_data.update({init_state: action})
        client.all_states[init_state.value[0]].append(action)
        if(not thread.is_alive()):
            lock.acquire()
            np.save(client.output_file, client.q_values)
            lock.release()
            conn.close()
            print(client.driver_name, "Disconnected.\n")
            main()
        send_action(conn, action, init_state)
        for j in range(0, vector_size): # generic response time
            state = State(client)
            state_vector.append(state)
            time.sleep(sampling_rate)
            next_state = State(client)
            state_vector.append(next_state)
            if(is_safe(next_state.value) and not is_safe(init_state.value) and j > 1):
                client.num_corrections += 1
                break
        # Q-learning lookup table update
        iteration_rewards = define_rewards(client, init_state, action, state_vector[-1])
        final_state = state_vector[-1]
        delta = step_size * (iteration_rewards + gamma * np.max(client.q_values[final_state.value[0], final_state.value[1], final_state.value[2], :]) - client.q_values[init_state.value[0], init_state.value[1], init_state.value[2], action])
        client.q_values[init_state.value[0], init_state.value[1], init_state.value[2], action] += delta
        lock.acquire()
        np.save(client.output_file, client.q_values) # save on each iteration
        lock.release()
        init_state = final_state
    client.block_thread = True
    plot(client, plot_data, episode)
    save_plot_data(client, plot_data)
    client.block_thread = False
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

# data management functions
def receive_data(client, conn):
    while True:
        try:
            data = conn.recv(4096)
            if(not client.block_thread):
                client.ldw_data = pickle.loads(data)
        except pickle.UnpicklingError:
            continue
        except ValueError:
            continue
        except ConnectionResetError:
            client.conn_reset = True
            lock.acquire()
            np.save(client.output_file, client.q_values) # custom file
            lock.release()
            break
        except EOFError:
            client.conn_reset = True
            lock.acquire()
            np.save(client.output_file, client.q_values) # custom file
            lock.release()
            break
        except BrokenPipeError:
            client.conn_reset = True
            lock.acquire()
            np.save(client.output_file, client.q_values) # custom file
            lock.release()
            break
def receive_human_state(client, input_file):
    lock.acquire()
    with open(input_file) as file:
        try:
            data = json.load(file)
            client.human_state = data.get(list(data)[-1])
            lock.release()
            return client.human_state
        except ValueError:
            lock.release()
            return client.human_state # return last value

# helper functions
def parse_arguments():
    argparser = argparse.ArgumentParser(
        description='Q-learning LDW Server')
    argparser.add_argument(
        '-n', '--hostname',
        metavar='HOSTNAME',
        default='localhost',
        help='computer hostname or IP address')
    args = argparser.parse_args()
    return args
def save_plot_data(client, plot_data):
    lock.acquire()
    if(not(os.path.exists('Data'))):
          os.mkdir('Data')
    os.chdir('Data')
    driver_name = client.driver_name
    if(not(os.path.exists(driver_name))):
        os.mkdir(driver_name)
    os.chdir(driver_name)
    iterations_data_file_name = driver_name + '_iterations_data.json'
    wf_data_file_name = driver_name + '_warning_frequency_data.json'
    plot_data_values = {}
    for state in plot_data:
        if(str(state.value) in plot_data_values):
            value = plot_data_values[str(state.value)]
            value.append(int(plot_data[state]))
            plot_data_values.update({str(state.value): value})
        else:
            plot_data_values.update({str(state.value): [int(plot_data[state])]})
    dt = datetime.now()
    timestamp = dt.strftime('%d-%b-%Y (%H:%M)')
    iterations_data = {timestamp: plot_data_values}
    if(not os.path.exists(iterations_data_file_name)):
        with open(iterations_data_file_name, 'w') as file:
            json.dump(iterations_data, file)
    else:
        with open(iterations_data_file_name, 'r') as file:
            data = json.load(file)
            data.update(iterations_data)
            with open(iterations_data_file_name, 'w') as file:
                json.dump(data, file)
    with open(wf_data_file_name, 'w') as file:
        json.dump(client.state_counts, file)
    os.chdir('../..')
    lock.release()
def plot(client, plot_data, episode):
    lock.acquire()
    state_counts = client.state_counts
    driver_name = client.driver_name
    # manage directories
    if(not(os.path.exists('Plots'))):
          os.mkdir('Plots')
    os.chdir('Plots')
    if(not(os.path.exists(driver_name))):
        os.mkdir(driver_name)
    os.chdir(driver_name)
    # distance/iteration plot
    plt.axis([0, iterations, 11, 0])
    dt = datetime.now()
    timestamp = dt.strftime('%d-%b-%Y (%H:%M)')
    timestamp_alt = dt.strftime('%d-%b-%Y_%H%M')
    file_name = driver_name + '_' + timestamp_alt + '_ep' + str(episode) + '.png'
    title = driver_name + " plot for episode " + str(episode) + " on " + timestamp
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Distance state (lower is safer)")
    i = 1
    for key in plot_data:
        y = key.value[0]
        if(plot_data[key] == warning):
            plt.plot(i, y, 'ro') # plot (iteration, state) warning as red
        else:
            plt.plot(i, y, 'go') # plot (iteration, state) no warning as green
        i += 1
    plt.savefig(file_name, dpi=600)
    plt.clf()
    # warning frequency plot
    for i in range(3, 9): # initialization
        state_counts[i].append(0) # add zero entry for this episode
    episode = len(state_counts[3])
    for key in plot_data:
        if(plot_data[key] == warning and key.value[0] >= 3 and key.value[0] <= 8):
            state_counts[key.value[0]][episode-1] += 1
    timestamp = dt.strftime('%d-%b-%Y')
    file_name = driver_name + '_warning_frequency_plot_' + timestamp + '.png'
    title = driver_name + " warning frequency plot"
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Number of warnings")
    plt.axis([0, episode+1, 0, 10])
    plt.xticks(range(1, episode+1))
    plt.yticks(range(0, 10))
    state_colors = {3: '#9EC384', 4: '#BBD5AB', 5: '#FAE6A2', 6: '#F9DB79', 7: '#DE9C9A', 8: '#D16D69'}
    #ax = plt.subplot(111)
    width = 0.1
    for x in range(1, episode+1):
        offset = 0.3
        for key in state_counts:
            y = state_counts[key][x-1]
            plt.bar(x-offset, y, width = width, color = state_colors[key], align = 'center')
            offset -= 0.1
    labels = list(state_colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=state_colors[label]) for label in labels]
    plt.legend(handles, labels, title = "Distance state number (lower is safer)", loc='best')
    plt.savefig(file_name, dpi=600)
    plt.clf()
    client.state_counts = state_counts
    os.chdir('../..')
    lock.release()
def generate_statistics(client, episode, time_elapsed):
    lock.acquire()
    # manage directories
    if(not(os.path.exists('Statistics'))):
          os.mkdir('Statistics')
    os.chdir('Statistics')
    if(not(os.path.exists(client.driver_name))):
        os.mkdir(client.driver_name)
    os.chdir(client.driver_name)
    # get data
    warning_states = client.warning_states
    all_states = client.all_states
    warning_state_values = []
    warning_ratios = {}
    dr = []
    dl = []
    if(len(warning_states) == 0):
        return
    for state in warning_states:
        warning_state_values.append(str(state.value))
        dr.append(state.metrics.get("dr"))
        dl.append(state.metrics.get("dl"))
    for dist_state in all_states:
        num_warnings = 0
        if(len(all_states[dist_state]) == 0):
            warning_ratios.update({dist_state: "N/A"})
            continue
        for action in all_states[dist_state]:
            if(action == warning):
                num_warnings += 1
        ratio = "{0:.1%}".format(num_warnings/len(all_states[dist_state]))
        warning_ratios.update({dist_state: ratio})
    try:
        most_common_state = statistics.mode(warning_state_values)
    except statistics.StatisticsError:
        most_common_state = warning_state_values[-1]
    avg_dr = statistics.mean(dr)
    avg_dl = statistics.mean(dl)
    total_time_run = convert(time_elapsed)
    data = {"q_table_name": client.output_file, "driver_id": client.driver_id, "warning_most_common_state": most_common_state, "avg_warning_dr": avg_dr, "avg_warning_dl": avg_dl, "total_time_run": total_time_run, "total_time_run_seconds": time_elapsed, "total_num_episodes": episode, "num_corrections": client.num_corrections, "num_invasions": client.num_invasions, "num_warning_states": len(warning_states), "warning_ratio_dist_states": warning_ratios}
    statistics_file = client.statistics_file
    write_statistics(data, statistics_file)
    lock.release()
def write_statistics(data, statistics_file):
    if(not os.path.exists(statistics_file)):
        with open(statistics_file, 'w') as file:
            json.dump(data, file, indent = 4)
    else:
        with open(statistics_file) as file:
            old_data = json.load(file)
            old_data_percentage = old_data["num_warning_states"] / (old_data["num_warning_states"] + data["num_warning_states"])
            new_data_percentage = 1 - old_data_percentage
            data["avg_warning_dr"] = (new_data_percentage * data["avg_warning_dr"]) + (old_data_percentage * old_data["avg_warning_dr"])
            data["avg_warning_dl"] = (new_data_percentage * data["avg_warning_dl"]) + (old_data_percentage * old_data["avg_warning_dl"])
            data["total_time_run_seconds"] += old_data["total_time_run_seconds"]
            data["total_time_run"] = convert(data["total_time_run_seconds"])
            data["total_num_episodes"] = old_data["total_num_episodes"] + 1
            data["num_corrections"] += old_data["num_corrections"]
            data["num_invasions"] += old_data["num_invasions"]
            data["num_warning_states"] += old_data["num_warning_states"]
            for dist_state in old_data["warning_ratio_dist_states"]:
                if(old_data["warning_ratio_dist_states"][dist_state] == "N/A"):
                    continue # sets data to new value
                if(data["warning_ratio_dist_states"][int(dist_state)] == "N/A"):
                    data["warning_ratio_dist_states"][int(dist_state)] == old_data["warning_ratio_dist_states"][dist_state] # data does not change
                    continue
                data["warning_ratio_dist_states"][int(dist_state)] = "{0:.1%}".format((new_data_percentage * float(data["warning_ratio_dist_states"][int(dist_state)].strip('%')) + (old_data_percentage * float(old_data["warning_ratio_dist_states"][dist_state].strip('%'))))/100)
            if(old_data.get("driver_id") is not None):
                data["driver_id"] = old_data["driver_id"]
            file.close()
            with open(statistics_file, 'w') as file:
                json.dump(data, file, indent = 4)
    os.chdir('../..')
def load_state_counts(client): # state counts need to be loaded to form warning frequency plot of all data, iteration data is write-only
    lock.acquire()
    data_file = client.driver_name + "_warning_frequency_data.json"
    data_path = 'Data/' + client.driver_name + '/' + data_file
    state_counts = {3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    if(os.path.exists(data_path)):
        os.chdir('Data/' + client.driver_name)
        with open(data_file) as file:
            data = json.load(file)
            i = 3
            for key in data:
                state_counts[i] = data[key] # load in state_counts dictionary if it already exists for plotting frequency graphs
                i += 1
            file.close()
        os.chdir('../..')
    lock.release()
    return state_counts
def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return '%d:%02d:%02d' % (hour, minutes, seconds)

# driving functions
def main():
    global lock
    args = parse_arguments()
    hostname_to_IP = {'iMac': '192.168.0.5', 'MBP': '192.168.0.78', 'MBPo': '192.168.254.41', 'localhost': '127.0.0.1'}
    IP = hostname_to_IP.get(args.hostname)
    if(IP is None):
        IP = args.hostname
    port = 50007
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((IP, port))
    lock = threading.Lock()
    while True:
        try:
            sock.listen(1)
            conn, addr = sock.accept()
            thread = threading.Thread(target=client_thread, args=(conn,))
            thread.start()
        except KeyboardInterrupt:
            sock.close()
            exit()
def client_thread(conn):
    driver_name = None
    while True:
        data = conn.recv(4096)
        if(data.decode() is not None):
            driver_name = data.decode()
            conn.send("Success".encode())
            break
    print("Connected by", driver_name)
    client = client_class.Client(driver_name)
    client.state_counts = load_state_counts(client)
    thread = threading.Thread(target=receive_data, args=(client, conn, ))
    thread.start()
    server_loop(client, conn)
def server_loop(client, conn):
    thread = threading.currentThread()
    episode = 1
    init_time = time.time()
    try:
        while(True):
            try:
                print("Running episode", episode, "for", client.driver_name)
                q_learning(client, conn, thread, episode)
                if(client.statistics_file is not None):
                    time_elapsed = time.time() - init_time
                    init_time = time.time()
                    generate_statistics(client, episode, time_elapsed)
                    client.warning_states = []
                    client.num_corrections = 0
                    client.num_invasions = 0
                episode += 1
                print("Finished episode", episode, "for", client.driver_name)
            except BrokenPipeError:
                print(client.driver_name, "Disconnected.\n")
                break
    except EOFError:
        conn.close()
        thread.join()
    except KeyboardInterrupt:
        if(client.statistics_file is not None):
            time_elapsed = time.time() - init_time
            init_time = time.time()
            generate_statistics(client, episode, time_elapsed)
        conn.close()
        thread.join()
main()
