import numpy as np
import os
import time
if(os.path.exists("DriverQValues.npy")):
    while(True):
        print("Q-table:")
        q_values = np.load("DriverQValues.npy")
        np.set_printoptions(suppress=True)
        print(q_values)
        time.sleep(2)
        os.system('clear')
        print("")
else:
    print("DriverQValues.npy does not exist!")
