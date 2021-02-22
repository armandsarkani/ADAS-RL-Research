import json
import argparse
import os
import numpy as np
from sklearn.feature_selection import f_regression, mutual_info_regression, mutual_info_classif
from sklearn.metrics import mutual_info_score

def main():
    # initialization
    argparser = argparse.ArgumentParser(
        description='Mutual Information')
    argparser.add_argument(
        '-dn', '--name',
        metavar='DRIVER_NAME',
        default='AggressiveDriver',
        help='specify a driver name for whom to load datasets')
    args = argparser.parse_args()
    driver_name = args.name
    if(not(os.path.exists('Data'))):
        os.mkdir('Data')
    os.chdir('Data')
    if(not(os.path.exists(driver_name))):
        os.mkdir(driver_name)
    os.chdir(driver_name)
    driver_file_name = driver_name + '_mi_data.json'
    driver_file = open(driver_file_name, 'r')
    data = json.load(driver_file)
    # creating disjoint datasets
    states = []
    actions = []
    vector_sizes = []
    for element in data:
        states.append(element['state'][0] + element['state'][1] + element['state'][2])
        actions.append(element['action'])
        vector_sizes.append(element['vector_size'])
    states = np.array(states)
    actions = np.array(actions)
    vector_sizes = np.array(vector_sizes)
    mi = mutual_info_score(actions, states)
    print(mi)
    #print(mutual_info_classif(states.reshape(-1,1), actions, discrete_features = True)) # mutual information of 0.69, expressed in nats
    os.chdir('..')
main()
