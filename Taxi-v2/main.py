from agent import Agent
from pyeasyga import GeneticAlgorithm
import time
#from pyeasyga.pyeasyga import GeneticAlgorithm
from monitor import interact
import gym
import numpy as np
import random

env = gym.make('Taxi-v3')

def interact_function(epsilon, gamma, epsilonreducer):
    agent = Agent(epsilon=epsilon, gamma=gamma, epsilonreducer=epsilonreducer)
    avg_rewards, best_avg_reward = interact(env, agent)
    return -(best_avg_reward)                                 ## Because i need to maximise this !!


data = { 'epsilon' : [0.01, 0.6], 'gamma' : [0.5, 1.0], 'epsilonreducer' : [1.001, 100] }

def create_individual(data):
    paramsValue = {};
    paramsValue['epsilon'] = round(random.uniform(data['epsilon'][0], data['epsilon'][1]), 5)
    paramsValue['gamma'] = round(random.uniform(data['gamma'][0], data['gamma'][1]), 4)
    paramsValue['epsilonreducer'] = round(random.uniform(data['epsilonreducer'][0], data['epsilonreducer'][1]), 2)
    return paramsValue

def fitness_function (individual, parameter):
    agent = Agent(epsilon=individual['epsilon'], gamma=individual['gamma'], epsilonreducer=individual['epsilonreducer'])
    avg_rewards, best_avg_reward = interact(env, agent)
    return best_avg_reward 

def crossover_function(parent_1, parent_2):
    num = random.choice([1, 2, 3])
    child1 = {}
    child2 = {}
    if num == 1 :
        child1['epsilon'] = parent_2['epsilon']
        child2['epsilon'] = parent_1['epsilon']
        child1['gamma'] = parent_2['gamma']
        child2['gamma'] = parent_1['gamma']
        child1['epsilonreducer'] = parent_1['epsilonreducer']
        child2['epsilonreducer'] = parent_2['epsilonreducer']
    elif num == 2:
        child1['epsilon'] = parent_1['epsilon']
        child2['epsilon'] = parent_2['epsilon']
        child1['gamma'] = parent_2['gamma']
        child2['gamma'] = parent_1['gamma']
        child1['epsilonreducer'] = parent_2['epsilonreducer']
        child2['epsilonreducer'] = parent_1['epsilonreducer']
    elif num == 3:
        child1['epsilon'] = parent_2['epsilon']
        child2['epsilon'] = parent_1['epsilon']
        child1['gamma'] = parent_1['gamma']
        child2['gamma'] = parent_2['gamma']
        child1['epsilonreducer'] = parent_2['epsilonreducer']
        child2['epsilonreducer'] = parent_1['epsilonreducer']
    return child1, child2

def mutate(individual):
    num = random.choice([1, 2, 3])
    if num == 1:
        individual['epsilon'] = round(random.uniform(data['epsilon'][0], data['epsilon'][1]), 5)
    elif num == 2:
        individual['gamma'] = round(random.uniform(data['gamma'][0], data['gamma'][1]), 4)
    elif num == 3:
        individual['epsilonreducer'] = round(random.uniform(data['epsilonreducer'][0], data['epsilonreducer'][1]), 2)

ga = GeneticAlgorithm(data,
                      population_size=10,
                      generations=2,
                      crossover_probability=0.8,
                      mutation_probability=0.05,
                      elitism=True,
                      maximise_fitness=True)

ga.fitness_function = fitness_function
ga.create_individual = create_individual
ga.crossover_function = crossover_function
ga.mutate_function = mutate

#ga.run(n_workers=10, parallel_type="threading")
tic = time.perf_counter()
ga.run(n_workers=10, parallel_type="process")
toc = time.perf_counter()

print (f"GA Finished running in {toc-tic:0.4f} seconds")


