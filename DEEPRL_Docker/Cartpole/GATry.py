import numpy as np

import random

from collections import deque

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import gym 
from agent import Agent

import torch

from deap import algorithms

env = gym.make('CartPole-v1')

pop_size = 50  
sigma = 0.5   ##sigma (float): standard deviation of additive noise
gamma = 1
max_t = 1000
elite_frac = 0.2
n_elite = int(elite_frac * pop_size)
n_generations = 20
CXPB = 0.2
MUTPB = 0.5

agent = Agent(env)

best_Weights = {}

best_Weights['fc1_W'] = sigma * np.random.randn(agent.getfc1_W_size())
best_Weights['fc1_b'] = sigma * np.random.randn(agent.getfc1_b_size())
best_Weights['fc2_W'] = sigma * np.random.randn(agent.getfc2_W_size())
best_Weights['fc2_b'] = sigma * np.random.randn(agent.getfc2_b_size())

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", dict, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def create_weights(best_Weights):
    weight ={}
    weight['fc1_W'] = best_Weights['fc1_W'] + (sigma * np.random.randn(agent.getfc1_W_size()))
    weight['fc1_b'] = best_Weights['fc1_b'] + (sigma * np.random.randn(agent.getfc1_b_size()))
    weight['fc2_W'] = best_Weights['fc2_W'] + (sigma * np.random.randn(agent.getfc2_W_size()))
    weight['fc2_b'] = best_Weights['fc2_b'] + (sigma * np.random.randn(agent.getfc2_b_size()))
    return weight

toolbox.register("attr_weights", create_weights, best_Weights)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_weights)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluation(individual):
    reward = agent.evaluate(individual, gamma, max_t)
    return reward,

toolbox.register("evaluate", evaluation)

def crossover(ind1, ind2):
    ind2['fc1_W'] = ind1['fc1_W']
    ind2['fc1_b'] = ind1['fc1_b']
    ind1['fc2_W'] = ind2['fc2_W']
    ind1['fc2_b'] = ind2['fc2_b']
    return ind1, ind2

toolbox.register("mate", crossover)

def mutate(individual):
    num = random.choice([1,2])
    if num == 1 :
        individual['fc1_W'] = best_Weights['fc1_W'] + (sigma * np.random.randn(agent.getfc1_W_size()))
        individual['fc1_b'] = best_Weights['fc1_b'] + (sigma * np.random.randn(agent.getfc1_b_size()))
    if num == 2 :
        individual['fc2_W'] = best_Weights['fc2_W'] + (sigma * np.random.randn(agent.getfc2_W_size()))
        individual['fc2_b'] = best_Weights['fc2_b'] + (sigma * np.random.randn(agent.getfc2_b_size()))
    return individual,

toolbox.register("mutate", mutate)
toolbox.register("select",tools.selBest,k = n_elite)

pop = toolbox.population(n = pop_size)

invalid_ind = [ind for ind in pop if not ind.fitness.valid]
fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
for ind, fit in zip(invalid_ind, fitnesses):
    ind.fitness.values = fit

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

logbook = tools.Logbook()
logbook.header = "gen", "evals", "std", "min", "avg", "max"

record = stats.compile(pop)
logbook.record(gen=0, evals=len(pop), **record)
print(logbook.stream)

scores_deque = deque(maxlen=100)
scores = []

for gen in range(1, n_generations):
    pop = toolbox.select(pop, k = n_elite)
    reward = agent.evaluate(pop[0], gamma=1.0)
    best_Weights['fc1_W'] = np.array([pops['fc1_W'] for pops in pop]).mean(axis=0)
    best_Weights['fc1_b'] = np.array([pops['fc1_b'] for pops in pop]).mean(axis=0)
    best_Weights['fc2_W'] = np.array([pops['fc2_W'] for pops in pop]).mean(axis=0)
    best_Weights['fc2_b'] = np.array([pops['fc2_b'] for pops in pop]).mean(axis=0)
    offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    pop = pop + offspring
    new_pop = toolbox.population(n = (pop_size - len(pop)))
    invalid_ind = [ind for ind in new_pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    pop += new_pop
    record = stats.compile(pop)
    logbook.record(gen=gen, evals=len(pop), **record)
    print(logbook.stream)
    if record["avg"]>=90.0:
        print('\nEnvironment solved in {:d} generations!'.format(gen))
        agent.evaluate(pop[0], gamma=1.0)
        torch.save(agent.state_dict(), 'checkpointCartPole.pth')
        break


