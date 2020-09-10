# **Privacy-Aware Personalized ADAS Research**
UCI undergraduate research project exploring personalized Advanced Driver Assistance Systems (ADAS) that are privacy-aware and adaptive based on various human states. <br/>

## Information

This repository mainly includes code written in Python and used with CARLA Simulator, an open-source driving simulation environment. The CARLA Python API files (for CARLA 0.9.5) are also included in order to make this code run on a CARLA build. They are located in the **carla** directory.

The primary objective up until now has been the Lane Departure Warning sensor, which employs a Q-learning reinforcement learning algorithm to continuously improve the accuracy of the sensor depending on the particular user's driving habits and their human state metrics (such as sleep and readiness). 

The aforementioned human state metrics are measured using the [Oura Ring] (https://ouraring.com). 

Relevant code for this objective is located in the **Lane Departure Warning** directory. That directory also includes relevant debugging logs and input/output files relating to Q-learning lookup tables, human state data, etc.

## Running Scripts

Once the repository is cloned, and CARLA is properly installed on your computer (Windows or Linux), you can run the Q-learning server (located in the **Lane Departure Warning** directory) in the following way:

    python q_learning_server.py --hostname HOSTNAME --input INPUT.json --output OUTPUT.npy

The hostname or IP address of your computer is needed to establish a connection with a client. By default, localhost (127.0.0.1) is used. The input file in this case is an existing JSON file containing human state data from the Oura Ring. If you do not specify this argument, the default file name (HumanStates.json) will be used. An input file must exist to run the server (see below). The output file is the name of the NumPy file where the driver's Q-learning lookup table is located. The server will continuously output data to this file by updating the existing Q-table. If this file does not exist, the server will automatically create it for you and initialize the Q-table on the first run. If you do not specify this argument, the default file name (DriverQValues.npy) will be used.

To receive data from the Oura Ring, please run the following script:

    python oura_server.py --output OUTPUT.json

The output file refers to the JSON file where you wish to output the records of human state values. If the file does not exist, it will be created for you. If you do not specify this argument, the default file name (HumanStates.json) will be used. Currently, the Personal Access Token needed to access any data from Oura points to my personal ring. If you wish to use your own Personal Access Token, please modify the code.

Once the server is up and running, a client program (such as the driving scenarios within q_learning_client.py) can be run in the following way:

    python q_learning_client.py --hostname HOSTNAME --mode MODE --throttle #.# --driver DRIVER

The hostname argument must be exactly the same as the server program to establish a connection. The mode argument specifies whether or not you wish to load the CARLA world needed for this simulation (Town06) to run as intended. A value of "set" loads the CARLA world. The mode argument only needs to be set on the first run (unless the CARLA Simulator is relaunched), and can be omitted in future runs. The throttle argument specifies the constant vehicle throttle value (between 0.0 and 1.0) to be used in this simulation. The driver argument indicates the category of response time you wish to have for this particular driver simulation. Currently, the three available options are "fast", "slow", and "cautious".

