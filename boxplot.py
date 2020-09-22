import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
import math

gain = []
for i in range(10):
    experiment_name = f'En3_select_best_{i}'
    with open(experiment_name+'/gain.csv') as input:

        reader = csv.reader(input, delimiter=' ')
        next(reader)

        p_energy = []
        e_energy = []
        for rows in reader:
            p_health = float(rows[1])
            e_health = float(rows[2])
            gain.append(p_health - e_health)

fig = plt.figure()
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
ax1 = fig.add_subplot(1,1,1)
ax1.boxplot(gain)
plt.show()
# ax2 = fig.add_subplot(1,2,2)
