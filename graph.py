import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
import math


n_islands = 5
gen = 20

pops_m_best = {}
pops_b_best = {}
pops_s_best = {}

for i in range(10):
    experiment_name = f'En1_select_best_{i}'
    with open(experiment_name+'/results.csv') as input:

        reader = csv.reader(input, delimiter=' ')
        next(reader)

        row_n = 0
        row_prev_gen = 0
        b_best = -6
        for rows in reader:
            if int(rows[0]) in pops_m_best.keys():
                pops_m_best[int(rows[0])].append(float(rows[3]))
            else:
                pops_m_best[int(rows[0])] = [float(rows[3])]
            if row_prev_gen != int(rows[0]) or row_n == 99:
                if row_prev_gen in pops_b_best.keys():
                    pops_b_best[row_prev_gen].append(b_best)
                else:
                    pops_b_best[row_prev_gen] = [b_best]
                b_best = -6
                row_prev_gen += 1
            if row_prev_gen == int(rows[0]) and float(rows[2]) > b_best:
                b_best = float(rows[2])
            if int(rows[0]) in pops_s_best.keys():
                pops_s_best[int(rows[0])].append(float(rows[4]))
            else:
                pops_s_best[int(rows[0])] = [float(rows[4])]
            row_n += 1

mean_b_best = []
std_b_best = []
mean_m_best = []
std_s_best = []
for j in range(gen):
    mean_b_best.append(np.mean(pops_b_best[j]))
    std_b_best.append(np.std(pops_b_best[j]))
    mean_m_best.append(np.mean(pops_m_best[j]))
    sq_std = list(map(lambda x: x**2, pops_s_best[j]))
    std = math.sqrt(np.mean(sq_std))
    std_s_best.append(std)


fig = plt.figure()
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
x = np.linspace(0,19,20)
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.errorbar(x, mean_b_best, std_b_best, linestyle='None', label="Average best fitness", marker="o", capsize=5)
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness")
ax1.set_xticks(np.arange(0,20,1))
ax1.legend(loc="best")
ax1.set_title("Average best fitness for enemy 1")

ax2.errorbar(x, mean_m_best, std_s_best, linestyle='None', label="Average fitness", marker="o", capsize=5)
ax2.set_xlabel("Generation")
ax2.set_ylabel("Fitness")
ax2.set_xticks(np.arange(0,20,1))
ax2.legend(loc="best")
ax2.set_title("Average fitness for enemy 1")
# for i in range(n_islands):
#     ax1.plot(x, pops_m[i])
#     ax2.plot(x, pops_b[i])

# for i in range(n_islands):
#     values = pops[i][9:].reshape((16,16))
#     x = np.linspace(0,15,16)
#     y = np.linspace(0,15,16)
#     x, y = np.meshgrid(x,y)
#     ax = fig.add_subplot(1,n_islands, i+1, projection='3d')
#     ax.plot_surface(x,y,values,cmap=plt.cm.coolwarm)

plt.show()
