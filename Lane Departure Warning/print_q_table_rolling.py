import numpy as np
import os
import time
import argparse
argparser = argparse.ArgumentParser(
description='Print Q-table continuously')
argparser.add_argument(
    '-i', '--input',
    metavar='INPUT.npy',
    default='DriverQValues.npy',
    help='specify an input NumPy file')
args = argparser.parse_args()
input_file = args.input
if(os.path.exists(input_file)):
    while(True):
        print("Q-table:")
        q_values = np.load(input_file)
        np.set_printoptions(suppress=True)
        print(q_values)
        time.sleep(2)
        os.system('clear')
        print("")
else:
    print(input_file + " does not exist!")
