import numpy as np
import os
if(os.path.exists("DriverQValues.npy")):
    q_values = np.load("DriverQValues.npy")
    np.set_printoptions(suppress=True)
    print(q_values)
else:
    print("DriverQValues.npy does not exist!")
