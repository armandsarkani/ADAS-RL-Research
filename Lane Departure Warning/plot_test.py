import matplotlib.pyplot as plt
import json
import sys
import os
def load_state_counts():
    data_file = 'FastDriver_warning_frequency_data.json'
    data_path = 'Data/FastDriver/' + data_file
    state_counts = {3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    if(os.path.exists(data_path)):
        os.chdir('Data/FastDriver')
        with open(data_file) as file:
            data = json.load(file)
            i = 3
            for key in data:
                state_counts[i] = data[key] # load in state_counts dictionary if it already exists for plotting frequency graphs
                i += 1
            file.close()
        os.chdir('../..')
    return state_counts
def plot(state_counts):
    driver_name = 'FastDriver'
    for i in range(3, 9): # initialization
        state_counts[i].append(0) # add zero entry for this episode
    episode = 28
    file_name = driver_name + '_warning_frequency_plot.png'
    plt.xlabel("Episodes")
    plt.ylabel("Number of warnings")
    plt.axis([0, episode+1, 0, 10])
    plt.xticks(range(1, episode+1), rotation = 90)
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

plot(load_state_counts())
