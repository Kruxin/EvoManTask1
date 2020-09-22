import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
import math


n_islands = 5
gen = 20
enemy = 1

pops_m_best = {}
pops_b_best = {}
pops_s_best = {}
pops_m_random = {}
pops_b_random = {}
pops_s_random = {}
pops_m_no = {}
pops_b_no = {}
pops_s_no = {}

for i in range(10):
    experiment_name1 = f'En{enemy}_select_best_{i}'
    experiment_name2 = f'En{enemy}_select_random_{i}'
    experiment_name3 = f'En{enemy}_no_isl_{i}'
    with open(experiment_name1+'/results.csv') as input:

        reader = csv.reader(input, delimiter=' ')
        next(reader)

        row_n1 = 0
        row_prev_gen1 = 0
        b_best1 = -6
        for rows in reader:
            if int(rows[0]) in pops_m_best.keys():
                pops_m_best[int(rows[0])].append(float(rows[3]))
            else:
                pops_m_best[int(rows[0])] = [float(rows[3])]
            if row_prev_gen1 != int(rows[0]) or row_n1 == 99:
                if row_prev_gen1 in pops_b_best.keys():
                    pops_b_best[row_prev_gen1].append(b_best1)
                else:
                    pops_b_best[row_prev_gen1] = [b_best1]
                b_best1 = -6
                row_prev_gen1 += 1
            if row_prev_gen1 == int(rows[0]) and float(rows[2]) > b_best1:
                b_best1 = float(rows[2])
            if int(rows[0]) in pops_s_best.keys():
                pops_s_best[int(rows[0])].append(float(rows[4]))
            else:
                pops_s_best[int(rows[0])] = [float(rows[4])]
            row_n1 += 1

    with open(experiment_name2+'/results.csv') as input:

        reader2 = csv.reader(input, delimiter=' ')
        next(reader2)

        row_n = 0
        row_prev_gen = 0
        b_best = -6
        for rows in reader2:
            if int(rows[0]) in pops_m_random.keys():
                pops_m_random[int(rows[0])].append(float(rows[3]))
            else:
                pops_m_random[int(rows[0])] = [float(rows[3])]
            if row_prev_gen != int(rows[0]) or row_n == 99:
                if row_prev_gen in pops_b_random.keys():
                    pops_b_random[row_prev_gen].append(b_best)
                else:
                    pops_b_random[row_prev_gen] = [b_best]
                b_best = -6
                row_prev_gen += 1
            if row_prev_gen == int(rows[0]) and float(rows[2]) > b_best:
                b_best = float(rows[2])
            if int(rows[0]) in pops_s_random.keys():
                pops_s_random[int(rows[0])].append(float(rows[4]))
            else:
                pops_s_random[int(rows[0])] = [float(rows[4])]
            row_n += 1

    with open(experiment_name3+'/results.csv') as input:

        reader3 = csv.reader(input, delimiter=' ')
        next(reader3)

        row_n2 = 1
        row_prev_gen2 = 1
        b_best2 = -6
        for rows in reader3:
            if int(rows[0]) in pops_m_no.keys():
                pops_m_no[int(rows[0])].append(float(rows[2]))
            else:
                pops_m_no[int(rows[0])] = [float(rows[2])]
            if int(rows[0]) in pops_b_no.keys():
                pops_b_no[int(rows[0])].append(float(rows[1]))
            else:
                pops_b_no[int(rows[0])] = [float(rows[1])]
            if int(rows[0]) in pops_s_no.keys():
                pops_s_no[int(rows[0])].append(float(rows[3]))
            else:
                pops_s_no[int(rows[0])] = [float(rows[3])]
            row_n2 += 1
            # print(row_n2)

# print(pops_b_best)


mean_b_best = []
std_b_best = []
mean_m_best = []
std_s_best = []
mean_b_random = []
std_b_random = []
mean_m_random = []
std_s_random = []
mean_b_no = []
std_b_no = []
mean_m_no = []
std_s_no = []
abs_best_best = []
abs_best_random = []
abs_best_no = []

for j in range(gen):
    if j != 0:
        mean_b_no.append(np.mean(pops_b_no[j]))
        std_b_no.append(np.std(pops_b_no[j]))
        mean_m_no.append(np.mean(pops_m_no[j]))
        sq_std_no = list(map(lambda z: z**2, pops_s_no[j]))
        std_no = math.sqrt(np.mean(sq_std_no))
        std_s_no.append(std_no)
        abs_best_no.append(np.max(pops_b_no[j]))

    mean_b_best.append(np.mean(pops_b_best[j]))
    mean_b_random.append(np.mean(pops_b_random[j]))

    std_b_best.append(np.std(pops_b_best[j]))
    std_b_random.append(np.std(pops_b_random[j]))

    mean_m_best.append(np.mean(pops_m_best[j]))
    mean_m_random.append(np.mean(pops_m_random[j]))

    sq_std = list(map(lambda x: x**2, pops_s_best[j]))
    sq_std_random = list(map(lambda y: y**2, pops_s_random[j]))

    std = math.sqrt(np.mean(sq_std))
    std_random = math.sqrt(np.mean(sq_std_random))

    std_s_best.append(std)
    std_s_random.append(std_random)

    abs_best_best.append(np.max(pops_b_best[j]))
    abs_best_random.append(np.max(pops_b_random[j]))



fig = plt.figure()
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
x = np.linspace(0,19,20)
x1 = np.linspace(1,19,19)
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
ax1.errorbar(x, mean_b_best, std_b_best, label="Best migration", marker="o", capsize=5)
ax1.errorbar(x, mean_b_random, std_b_random, label="Random migration", marker="o", capsize=5)
ax1.errorbar(x1, mean_b_no, std_b_no, label="No migration", marker="o", capsize=5)
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness")
ax1.set_xticks(np.arange(0,20,1))
ax1.legend(loc="best")
ax1.set_title(f"Average best fitness for enemy {enemy}")

ax2.errorbar(x, mean_m_best, std_s_best, label="Best migration", marker="o", capsize=5)
ax2.errorbar(x, mean_m_random, std_s_random, label="Random migration", marker="o", capsize=5)
ax2.errorbar(x1, mean_m_no, std_s_no, label="No migration", marker="o", capsize=5)
ax2.set_xlabel("Generation")
ax2.set_ylabel("Fitness")
ax2.set_xticks(np.arange(0,20,1))
ax2.legend(loc="best")
ax2.set_title(f"Average fitness for enemy {enemy}")

ax3.plot(x, abs_best_best, label="Best migration")
ax3.plot(x, abs_best_random, label="Random migration")
ax3.plot(x1, abs_best_no, label="No migration")
ax3.set_xlabel("Generation")
ax3.set_ylabel("Fitness")
ax3.set_xticks(np.arange(0,20,1))
ax3.legend(loc="best")
ax3.set_title(f"Best fitness for enemy {enemy}")
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
