###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import heapq
import random


experiment_name = 'migrate_swap_random'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

dom_u = 1
dom_l = -1
npop = 100
gens = 30
mutation = 0.2
last_best = 0
n_islands = 2
exch_rate = 1
n_migrate = 2


# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# normalizes
def norm(x, pfit_pop):

    if ( max(pfit_pop) - min(pfit_pop) ) > 0:
        x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
    else:
        x_norm = 0


    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm


# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

def scramble_pop(pop, fit_pop, fraction):
    """this function deletes a worst fraction of the population
    and replaces them with random ones, so as to keep diversity up"""

    ranked_worst = np.argsort(fit_pop)

    for rank in ranked_worst[:int(fraction * npop)]:
        for gene in range(0, len(pop[rank])-1):
            pop[rank][gene] = np.random.uniform(dom_l, dom_u, 1)

    return pop

def death_match(pop, fit_pop):
    """this function determines which individuals of the population get replaced"""

    # make some lists to keep track of individuals
    copy_pop = pop
    copy_fit_pop = fit_pop
    survivors_pop = []
    survivors_fitness = []
    death_match_pop = []
    death_match_fitness = []

    # we do 20 rounds of 5 randomly sampled individuals from the remaining pop
    for _ in range(int(npop / 5)):
        for _ in range(5):

            # choose the competitors
            if copy_pop.shape[0] > 1:
                index = np.random.randint(0, copy_pop.shape[0]-1, 1)[0]
            else:
                index = 0

            death_match_pop.append(copy_pop[index])
            death_match_fitness.append(copy_fit_pop[index])

            # delete the chosen ones from the population, so they can't be sampled again
            copy_pop = np.delete(copy_pop, [index], 0)
            copy_fit_pop = np.delete(copy_fit_pop, [index], 0)

        # now do the death match and add the best to the survivors
        survivor_index = np.argmax(death_match_fitness)
        survivors_pop.append(death_match_pop[survivor_index])
        survivors_fitness.append(death_match_fitness[survivor_index])

    # we return the 20 survivors and their fitness values as np arrays
    return np.array(survivors_pop), np.array(survivors_fitness)

def parent_selection(pop, fit_pop, rounds):
    """this function will select which 5 parents will mate"""

    # get the list of worst to best of the population
    worst_to_best = np.argsort(fit_pop)

    # select the parents based on which round, first 2 parents are sampled from top 40%
    p1 = pop[worst_to_best[pop.shape[0] - rounds - 1]]
    p2 = pop[worst_to_best[pop.shape[0] - rounds - 2]]

    # last 3 parents are randomly chosen
    p3, p4, p5 = pop[np.random.randint(0, pop.shape[0]-1, 3)]

    return np.array([p1, p2, p3, p4, p5])


def recombination(parents):
    """recombines 5 parents into 4 offspring"""

    # pick 5 random numbers that add up to 1
    random_values = np.random.dirichlet(np.ones(5),size=1)[0]

    # those random values will serve as weights for the genes 2 offspring get (whole arithmetic recombination)
    offspring1 = random_values[0] * parents[0] + random_values[1] * parents[1] + random_values[2] * parents[2] + random_values[3] * parents[3] + \
        random_values[4] * parents[4]

    # repeat for offspring 2
    random_values = np.random.dirichlet(np.ones(5),size=1)[0]
    offspring2 = random_values[0] * parents[0] + random_values[1] * parents[1] + random_values[2] * parents[2] + random_values[3] * parents[3] + \
        random_values[4] * parents[4]

    # the other 2 offspring will come from 4-point crossover
    random_points = np.sort(np.random.randint(1, parents[0].shape[0]-2, 4))

    # to make it so that it won't always be p1 who gives the first portion of DNA etc, we shuffle the parents
    np.random.shuffle(parents)

    # add the genes together
    offspring3 = np.concatenate((parents[0][0:random_points[0]], parents[1][random_points[0]:random_points[1]], parents[2][random_points[1]:random_points[2]],\
        parents[3][random_points[2]:random_points[3]], parents[4][random_points[3]:]))

    # repeat for offspring 4
    random_points = np.sort(np.random.randint(1, parents[0].shape[0]-2, 4))
    np.random.shuffle(parents)
    offspring4 = np.concatenate((parents[0][0:random_points[0]], parents[1][random_points[0]:random_points[1]], parents[2][random_points[1]:random_points[2]],\
        parents[3][random_points[2]:random_points[3]], parents[4][random_points[3]:]))

    # return the offspring
    return np.concatenate(([offspring1], [offspring2], [offspring3], [offspring4]))

def mutate(offspring):
    """this function will mutate the offspring"""

    # get the children and their genes
    offspring = offspring
    for child in offspring:

        # don't mutate every child, make it 50% of the offspring
        if np.random.uniform(0,0.4,1) < mutation:
            for gene in range(0, len(child)-1):

                # pick a random number between 0-1, mutate if < mutation rate
                if np.random.uniform(0,1,1) < mutation:

                    # change the gene by a small number from a very narrow normal distribution
                    child[gene] += np.random.normal(0, 0.2, 1)

                # make sure the genes don't get values outside of the limits
                if child[gene] > dom_u:
                    child[gene] = dom_u
                if child[gene] < dom_l:
                    child[gene] = dom_l

    return offspring

def select_random(pops, fit_pop):
    """
    Selects random indivuals for migration, creates dictionary for index, fitness and weights.
    """
    random_index_d = {}
    random_fit_inds_d = {}
    random_pop_inds_d = {}
    for i in range(n_islands):
        random_index_d[i] = random.sample(range(npop), n_migrate)
        random_fit_inds_d[i] = list(map(lambda x: fit_pop[i][x], random_index_d[i]))
        random_pop_inds_d[i] = list(map(lambda y: pops[i][y], random_index_d[i]))

    return random_index_d, random_fit_inds_d, random_pop_inds_d

def select_from_best_half(pops,fit_pop):
    """
    Selects best half of each population and saves n individuals from this selection for migration.
    Creates a new dictionary for index, fit values and weights.
    """
    index_d = {}
    fit_inds_d = {}
    pop_inds_d = {}
    for i in range(n_islands):
        index_d[i] = heapq.nlargest(int(npop/2), range(npop), key=lambda x: fit_pop[i][x])
        index_d[i] = random.sample(index_d[i], n_migrate)
        fit_inds_d[i] = list(map(lambda y: fit_pop[i][y], index_d[i]))
        pop_inds_d[i] = list(map(lambda z: pops[i][z], index_d[i]))

    return index_d, fit_inds_d, pop_inds_d

def select_best(pops, fit_pop):
    """
    Selects best n individuals from the population and saves it to a new dictionary,
    one for index, fit values and indivual weights.
    """
    best_index_d  = {}
    best_fit_inds_d = {}
    best_pop_inds_d = {}
    for i in range(n_islands):
        best_fit_inds_d[i] = heapq.nlargest(n_migrate, fit_pop[i])
        best_index_d[i] = heapq.nlargest(n_migrate, range(npop), key=lambda x: fit_pop[i][x])
        best_pop_inds_d[i] = list(map(lambda y: pops[i][y], best_index_d[i]))

    return best_index_d, best_fit_inds_d, best_pop_inds_d

def remove_best(best_index_d, pops, fit_pop):
    """
    Removes selection from current population and fitness dictionary. Returns updated dictionary.
    """
    for i in range(n_islands):
        best_index_d[i].sort(reverse=True)
        for j in range(n_migrate):
            pops[i] = np.delete(pops[i], best_index_d[i][j], axis=0)
            fit_pop_d[i] = np.delete(fit_pop_d[i], best_index_d[i][j], axis=0)

    return pops, fit_pop

def migrate(pops, fit_pop):
    """
    Selects the best n individuals from each island and replaces them for the best n from the next
    island in a circular pattern. Returns the updated dictionary for fitness and population.
    """
    best_index_d, best_fit_inds_d, best_pop_inds_d = select_from_best_half(pops, fit_pop)
    pops_mid, fit_pop_mid = remove_best(best_index_d, pops, fit_pop)
    for i in range(n_islands):
        for j in range(n_migrate):
            if i + 1 > n_islands - 1:
                pops[i] = np.vstack((pops[i], best_pop_inds_d[0][j]))
                fit_pop[i] = np.hstack((fit_pop[i], best_fit_inds_d[0][j]))
            else:
                pops[i] = np.vstack((pops[i], best_pop_inds_d[i+1][j]))
                fit_pop[i] = np.hstack((fit_pop[i], best_fit_inds_d[i+1][j]))

    return pops, fit_pop



# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)


# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')
    pops = {}
    fit_pop_d = {}
    best_d = {}
    mean_d = {}
    std_d = {}
    ini_g = 0
    solutions_d = {}
    for i in range(n_islands):
        pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
        pops[i] = pop
        fit_pop_d[i] = evaluate(pop)
        best_d[i] = np.argmax(fit_pop_d[i])
        mean_d[i] = np.mean(fit_pop_d[i])
        std_d[i] = np.std(fit_pop_d[i])
        solutions_d[i] = ([pop, fit_pop_d[i]])
    env.update_solutions(solutions_d)

else:

    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()

    pops = env.solutions[0]
    fit_pop_d = env.solutions[1]
    best_d = {}
    mean_d = {}
    std_d = {}

    for i in range(n_islands):
        best_d[i] = np.argmax(fit_pop_d[i])
        mean_d[i] = np.mean(fit_pop_d[i])
        std_d[i] = np.std(fit_pop_d[i])

    # finds last generation number
    file_aux  = open(experiment_name+'/gen.txt','r')
    ini_g = int(file_aux.readline())
    file_aux.close()


# saves results for first pop
file_aux  = open(experiment_name+'/results.csv','a')
if ini_g == 0:
    file_aux.write('gen island best mean std')
for i in range(n_islands):
    print( '\n GENERATION '+str(ini_g)+' '+str(i)+' '+str(round(fit_pop_d[i][best_d[i]], 6))+' '+str(round(mean_d[i],6))+' '+str(round(std_d[i],6)))
    file_aux.write('\n'+str(ini_g)+' '+str(i)+' '+str(round(fit_pop_d[i][best_d[i]],6))+' '+str(round(mean_d[i],6))+' '+str(round(std_d[i],6))   )
file_aux.close()


# evolution
new_best_counter_d = {}
all_time_best_d = {}
# last_sols_d = {}
# notimproved_d = {}

for i in range(n_islands):
    # last_sols_d[i] = fit_pop_d[i][best_d[i]]
    new_best_counter_d[i] = 0
    all_time_best_d[i] = 0

for i in range(ini_g+1, gens):

    if i % exch_rate == 0:
        pops, fit_pop_d = migrate(pops, fit_pop_d)
        print("\nMigration successfull\n")

    for j in range(n_islands):

        rounds = int(npop/5)
        offspring = np.zeros((0, n_vars))
        for k in range(1, rounds+1):

            # choose parents
            parents = parent_selection(pops[j], fit_pop_d[j], (k-1)*2)

            # honey, get the kids
            offspring_group = recombination(parents)

            # add them to the offspring array
            offspring = np.concatenate((offspring, offspring_group))

        # mutate half the offspring for diversity
        offspring = mutate(offspring)

        # we have the offspring, now we kill 80% of the population
        pops[j] = death_match(pops[j], fit_pop_d[j])[0]

        # mutate the surviving pop as well to increase search space
        pops[j] = mutate(pops[j])

        # combine the survivors with the offspring to form the new pop
        pops[j] = np.concatenate((pops[j], offspring))

        # test the pop
        fit_pop_d[j] = evaluate(pops[j])

        # get stats
        best_d[j] = np.argmax(fit_pop_d[j])
        std_d[j]  =  np.std(fit_pop_d[j])
        mean_d[j] = np.mean(fit_pop_d[j])

        # if 3 generations in a row don't give a new best solution, replace a fraction of the pop
        if fit_pop_d[j][best_d[j]] > all_time_best_d[j]:
            all_time_best_d[j] = fit_pop_d[j][best_d[j]]
            new_best_counter_d[j] = 0
            os.system(f"say 'New best is {round(all_time_best_d[j], 4)}' ")
        else:
            new_best_counter_d[j] += 1

        if new_best_counter_d[j] > 3:
            pops[j] = scramble_pop(pops[j], fit_pop_d[j], 0.3)
            new_best_counter_d[j] = 0

        # saves results
        file_aux  = open(experiment_name+'/results.csv','a')
        print( '\n GENERATION '+str(i)+' '+str(j)+' '+str(round(fit_pop_d[j][best_d[j]], 6))+' '+str(round(mean_d[j],6))+' '+str(round(std_d[j],6)))
        file_aux.write('\n'+str(i)+' '+str(j)+' '+str(round(fit_pop_d[j][best_d[j]],6))+' '+str(round(mean_d[j],6))+' '+str(round(std_d[j],6))   )
        file_aux.close()

    # Finds overall best individual
    best_fit = 0.0
    best_index = 0
    best_island = 0
    for j in range(n_islands):
        max_val = float(max(fit_pop_d[j]))
        if max_val > best_fit:
            best_fit = max_val
            best_index = np.argmax(fit_pop_d[j])
            best_island = j

    # saves file with the best solution
    np.savetxt(experiment_name+'/best.txt',pops[best_island][best_index])

    # saves simulation state
    solutions = [pops, fit_pop_d]
    env.update_solutions(solutions)
    env.save_state()

    # saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()




fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')


file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()


env.state_to_log() # checks environment state
