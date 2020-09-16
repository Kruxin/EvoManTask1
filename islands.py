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
npop = 10
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


# tournament
def tournament(pop, island):
    c1 =  np.random.randint(0,pop.shape[0], 1)
    c2 =  np.random.randint(0,pop.shape[0], 1)

    if fit_pop_d[island][c1] > fit_pop_d[island][c2]:
        return pop[c1][0]
    else:
        return pop[c2][0]


# limits
def limits(x):

    if x>dom_u:
        return dom_u
    elif x<dom_l:
        return dom_l
    else:
        return x


# crossover
def crossover(pop, island):

    total_offspring = np.zeros((0,n_vars))

    for p in range(0,pop.shape[0], 2):
        p1 = tournament(pop, island)
        p2 = tournament(pop, island)

        n_offspring =   np.random.randint(1,3+1, 1)[0]
        offspring =  np.zeros( (n_offspring, n_vars) )

        for f in range(0,n_offspring):

            cross_prop = np.random.uniform(0,1)
            offspring[f] = p1*cross_prop+p2*(1-cross_prop)

            # mutation
            for i in range(0,len(offspring[f])):
                if np.random.uniform(0 ,1)<=mutation:
                    offspring[f][i] =   offspring[f][i]+np.random.normal(0, 1)

            offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring



# kills the worst genomes, and replace with new best/random solutions
def doomsday(pop,fit_pop):

    worst = int(npop/4)  # a quarter of the population
    order = np.argsort(fit_pop)
    orderasc = order[0:worst]

    for o in orderasc:
        for j in range(0,n_vars):
            pro = np.random.uniform(0,1)
            if np.random.uniform(0,1)  <= pro:
                pop[o][j] = np.random.uniform(dom_l, dom_u) # random dna, uniform dist.
            else:
                pop[o][j] = pop[order[-1:]][0][j] # dna from best

        fit_pop[o]=evaluate([pop[o]])

    return pop,fit_pop

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
last_sols_d = {}
notimproved_d = {}
for i in range(n_islands):
    last_sols_d[i] = fit_pop_d[i][best_d[i]]
    notimproved_d[i] = 0

for i in range(ini_g+1, gens):

    if i % exch_rate == 0:
        pops, fit_pop_d = migrate(pops, fit_pop_d)
        print("\nMigration successfull\n")

    for j in range(n_islands):

        offspring = crossover(pops[j], j)  # crossover
        fit_offspring = evaluate(offspring)   # evaluation
        pops[j] = np.vstack((pops[j],offspring))
        fit_pop_d[j] = np.append(fit_pop_d[j],fit_offspring)

        print(fit_pop_d[j].shape)

        best_d[j] = np.argmax(fit_pop_d[j]) #best solution in generation
        fit_pop_d[j][best_d[j]] = float(evaluate(np.array([pops[j][best_d[j]] ]))[0]) # repeats best eval, for stability issues
        best_sol = fit_pop_d[j][best_d[j]]

        # selection
        fit_pop_cp = fit_pop_d[j]
        fit_pop_norm =  np.array(list(map(lambda y: norm(y,fit_pop_cp), fit_pop_d[j]))) # avoiding negative probabilities, as fitness is ranges from negative numbers
        probs = (fit_pop_norm)/(fit_pop_norm).sum()
        chosen = np.random.choice(pops[j].shape[0], npop , p=probs, replace=False)
        chosen = np.append(chosen[1:],best_d[j])
        pops[j] = pops[j][chosen]
        fit_pop_d[j] = fit_pop_d[j][chosen]


        # searching new areas

        if best_sol <= last_sols_d[j]:
            notimproved_d[j] += 1
        else:
            last_sols_d[j] = best_sol
            notimproved_d[j] = 0

        if notimproved_d[j] >= 15:

            print("\ndoomsday")
            # file_aux  = open(experiment_name+'/results.csv','a')
            # file_aux.write('doomsday')
            # file_aux.close()

            pops[j], fit_pop_d[j] = doomsday(pops[j],fit_pop_d[j])
            notimproved_d[j] = 0

        best_d[j] = np.argmax(fit_pop_d[j])
        std_d[j]  =  np.std(fit_pop_d[j])
        mean_d[j] = np.mean(fit_pop_d[j])

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
