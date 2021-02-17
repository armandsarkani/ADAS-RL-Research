import numpy as np
import os
import argparse
argparser = argparse.ArgumentParser(
description='Print Q-table once')
argparser.add_argument(
    '-i', '--input',
    metavar='INPUT.npy',
    default='DriverQValues.npy',
    help='specify an input NumPy file')
args = argparser.parse_args()
input_file = args.input
if(os.path.exists('Binaries/' + input_file)):
    q_values = np.load('Binaries/' + input_file)
    np.set_printoptions(suppress=True)
    print(q_values)
else:
    print(input_file + " does not exist!")
