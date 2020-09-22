################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

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



# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

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


# loads file with the best solution for testing

for g in range(10):
    experiment_name = f'En3_no_isl_{g}'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      enemies=[3],
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
    npop = 40
    gens = 20
    mutation = 0.2
    last_best = 0

    # initializes population loading old solutions or generating new ones

    if run_mode =='test':

        bsol = np.loadtxt(experiment_name+'/best.txt')
        print( '\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed','normal')
        evaluate([bsol])

        sys.exit(0)

    if not os.path.exists(experiment_name+'/evoman_solstate'):

        print( '\nNEW EVOLUTION\n')

        pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
        fit_pop = evaluate(pop)
        best = np.argmax(fit_pop)
        mean = np.mean(fit_pop)
        std = np.std(fit_pop)
        ini_g = 0
        solutions = [pop, fit_pop]
        env.update_solutions(solutions)

    else:

        print( '\nCONTINUING EVOLUTION\n')

        env.load_state()
        pop = env.solutions[0]
        fit_pop = env.solutions[1]

        best = np.argmax(fit_pop)
        mean = np.mean(fit_pop)
        std = np.std(fit_pop)

        # finds last generation number
        file_aux  = open(experiment_name+'/gen.txt','r')
        ini_g = int(file_aux.readline())
        file_aux.close()


    # saves results for first pop
    file_aux  = open(experiment_name+'/results.txt','a')
    file_aux.write('\n\ngen best mean std')
    print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()


    new_best_counter = 0
    all_time_best = 0

    for i in range(ini_g+1, gens):

        # HIER DE CODE VOOR DE EVO ALGOOOOO

        # recombination
        rounds = int(npop / 5)
        offspring = np.zeros((0, n_vars))
        for r in range(1, rounds+1):

            # choose parents
            parents = parent_selection(pop, fit_pop, (r-1)*2)

            # honey, get the kids
            offspring_group = recombination(parents)

            # add them to the offspring array
            offspring = np.concatenate((offspring, offspring_group))

        # mutate half the offspring for diversity
        offspring = mutate(offspring)

        # we have the offspring, now we kill 80% of the population
        pop = death_match(pop, fit_pop)[0]

        # mutate the surviving pop as well to increase search space
        pop = mutate(pop)

        # combine the survivors with the offspring to form the new pop
        pop = np.concatenate((pop, offspring))

        # test the pop
        fit_pop = evaluate(pop)

        # get stats
        best = np.argmax(fit_pop)
        std  =  np.std(fit_pop)
        mean = np.mean(fit_pop)

        # if 3 generations in a row don't give a new best solution, replace a fraction of the pop
        if fit_pop[best] > all_time_best:
            all_time_best = fit_pop[best]
            new_best_counter = 0
            os.system(f"say 'New best is {round(all_time_best, 4)}' ")
        else:
            new_best_counter += 1

        if new_best_counter > 3:
            pop = scramble_pop(pop, fit_pop, 0.3)
            new_best_counter = 0

        diversity_d = {}
        for k in range(npop):
            if fit_pop[k] in diversity_d.keys():
                diversity_d[fit_pop[k]] += 1
            else:
                diversity_d[fit_pop[k]] = 1
        file_aux = open(experiment_name+'/diversity.csv', 'a')
        file_aux.write('\n'+str(i)+' '+str(len(diversity_d.keys())))
        file_aux.close()

        ### FROM OPTIMIZE FILE ###

        # saves results
        file_aux  = open(experiment_name+'/results.csv','a')
        print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
        file_aux.close()

        # saves generation number
        file_aux  = open(experiment_name+'/gen.txt','w')
        file_aux.write(str(i))
        file_aux.close()

        # saves file with the best solution
        np.savetxt(experiment_name+'/best.txt',pop[best])

        # saves simulation state
        solutions = [pop, fit_pop]
        env.update_solutions(solutions)
        env.save_state()














fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')


file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()


env.state_to_log() # checks environment state
