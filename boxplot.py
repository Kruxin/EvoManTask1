import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
import math

enemy = 3

def read_data(experiment_name):
    gain = []
    with open(experiment_name+'/gain.csv') as input:

        reader = csv.reader(input, delimiter=' ')
        next(reader)

        p_energy = []
        e_energy = []
        for rows in reader:
            p_health = float(rows[1])
            e_health = float(rows[2])
            gain.append(p_health - e_health)
    return gain

gain_best_1 = []
gain_random_1 = []
gain_no_1 = []
gain_best_2 = []
gain_random_2 = []
gain_no_2 = []
gain_best_3 = []
gain_random_3 = []
gain_no_3 = []
for i in range(10):
    gain_best_1 = np.hstack((gain_best_1,read_data(f'En1_select_best_{i}')))
    gain_random_1 = np.hstack((gain_random_1,read_data(f'En1_select_random_{i}')))
    gain_no_1 = np.hstack((gain_no_1,read_data(f'En1_no_isl_{i}')))
    gain_best_2 = np.hstack((gain_best_2,read_data(f'En2_select_best_{i}')))
    gain_random_2 = np.hstack((gain_random_2,read_data(f'En2_select_random_{i}')))
    gain_no_2 = np.hstack((gain_no_2,read_data(f'En2_no_isl_{i}')))
    gain_best_3 = np.hstack((gain_best_3,read_data(f'En3_select_best_{i}')))
    gain_random_3 = np.hstack((gain_random_3,read_data(f'En3_select_random_{i}')))
    gain_no_3 = np.hstack((gain_no_3,read_data(f'En3_no_isl_{i}')))

ticks = [1,2,3]
labels = ["Best", "Random", "No"]

fig = plt.figure()
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
ax1 = fig.add_subplot(1,3,1)
ax1.boxplot([gain_best_1, gain_random_1, gain_no_1])
ax1.set_xticks(ticks, labels)
ax1.set_xticklabels(labels)
ax1.set_ylabel("Gain")
ax1.set_title(f"Enemy 1")
ax2 = fig.add_subplot(1,3,2)
ax2.boxplot([gain_best_2, gain_random_2, gain_no_2])
ax2.set_xticks(ticks, labels)
ax2.set_xticklabels(labels)
ax2.set_title(f"Enemy 2")
ax3 = fig.add_subplot(1,3,3)
ax3.boxplot([gain_best_3, gain_random_3, gain_no_3])
ax3.set_xticks(ticks, labels)
ax3.set_xticklabels(labels)
ax3.set_title(f"Enemy 3")
plt.show()
# ax2 = fig.add_subplot(1,2,2)
