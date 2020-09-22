import csv
import numpy as np
import sys
import matplotlib.pyplot as plt

def plotdiversity(enemy, method):
    """select enemy from 1-3, method = 'random' or 'best'"""

    # make dicts to save the data per gen for diversity, mean and std
    data_mean = {}
    data_std ={}

    for i in range(0, 10):

        with open(f'En{enemy}_select_{method}_{i}/diversity.csv') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            current_row = 1

            div = []
            div_all_gens = []
            std_all_gens = []
            for row in reader:
                if current_row > 1:
                    div.append(int(row[2]))

                if (current_row != 1) and ((current_row-1) % 5 == 0):
                    
                    div_all_gens.append(np.mean(div))
                    std_all_gens.append(np.std(div))
                    div = []
                current_row += 1

            data_mean[f'run{i}'] = div_all_gens
            data_std[f'run{i}'] = std_all_gens

    # now we get the actual lists to plot the data by taking the averages of the averages and std's

    y_data = []
    y_error = []
    x_data = np.linspace(1,20,19)
    #print(len(data_mean['run0']))
    #print(data_mean)
    print(data_std)
    for i in range(0, 19):
        y = np.mean([data_mean['run0'][i], data_mean['run1'][i], data_mean['run2'][i], data_mean['run3'][i], \
            data_mean['run4'][i], data_mean['run5'][i], data_mean['run6'][i], data_mean['run7'][i], data_mean['run8'][i], data_mean['run9'][i] ])
        y_data.append(y)
        y_err = np.sqrt(sum([data_std['run0'][i]**2, data_std['run1'][i]**2, data_std['run2'][i]**2, data_std['run3'][i]**2, \
            data_std['run4'][i]**2, data_std['run5'][i]**2, data_std['run6'][i]**2, data_std['run7'][i]**2, data_std['run8'][i]**2, data_std['run9'][i]**2 ])/19)
        y_error.append(y_err)


    fig, ax = plt.subplots()


    ax.errorbar(x_data, y_data,
                yerr=y_error,
                capsize=5,
                fmt='-o')


    ax.set_xlabel('Generation')
    ax.set_ylim(0, 30)
    ax.set_ylabel('Diversity')
    #ax.set_title('Enemy 1')


    # plt.show()

plotdiversity(1, 'random')
plotdiversity(1, 'best')
plt.show()
