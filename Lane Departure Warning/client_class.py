import glob
import os
import sys
import socket
import pickle
import math
import time
import threading
import numpy as np
import json
import logging
import statistics
import uuid
from datetime import datetime, date

# number of states per dimension
num_distance_states = 12
num_speed_states = 3
num_human_states = 3

class Client:
    driver_name = None
    input_file = None
    output_file = None
    statistics_file = None
    control = False
    epsilon = 0.1 # probability for exploration
    q_values = np.zeros((num_distance_states, num_speed_states, num_human_states, 2)) # q-value lookup table, initialized to zeros
    state_counts = {3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    driver_id = None
    human_state = 1
    warning_states = []
    all_states = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: []}
    num_corrections = 0
    num_invasions = 0
    sampling_rate = 0.1  # rate at which state transitions are measured
    vector_size = int(2/sampling_rate)
    min_vector_size = int(2/sampling_rate)/2
    max_vector_size = int(2/sampling_rate)*2
    block_thread = False
    ldw_data = None
    conn = None
    conn_reset = False
    def __init__(self, driver_name, conn):
        self.driver_name = driver_name
        self.conn = conn
        dt = datetime.now()
        timestamp = dt.strftime('%d%b')
        self.input_file = driver_name + 'HumanStates.json'
        self.output_file = driver_name + '.npy'
        self.statistics_file = driver_name + 'Statistics.json'
        if("Control" in driver_name or "control" in driver_name):
            self.control = True
        if(os.path.exists(self.output_file)):
            q_values = np.load(self.output_file)
            np.set_printoptions(suppress=True)
        else:
            np.save(self.output_file, self.q_values) # custom file
            self.initialize_q_table()
            epsilon = 0.15
        self.driver_id = uuid.uuid4().hex
    def initialize_q_table(self):
        for i in range(0, 3):
            self.q_values[i, :, :, 0] = 1
            self.q_values[i, :, :, 1] = 0
        for i in range(5, 8):
            self.q_values[i, :, :, 0] = 0
            self.q_values[i, :, :, 1] = 0
        for i in range(8, 12):
            self.q_values[i, :, :, 0] = 0
            self.q_values[i, :, :, 1] = 1
    def set_vector_size(self, size):
        new_size = int((self.vector_size + size)/2)
        if(new_size < self.min_vector_size):
            self.vector_size = self.min_vector_size
        elif(new_size > self.max_vector_size):
            self.vector_size = self.max_vector_size
        else:
            self.vector_size = new_size
