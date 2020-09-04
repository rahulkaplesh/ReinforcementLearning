import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, epsilon, gamma, epsilonreducer, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.epsilonreducer = epsilonreducer
        self.gamma = gamma
        self.alpha = 1
        print (f"AgentParams : Epsilon : { epsilon } : gamma : { gamma } : epsilonreducer : { epsilonreducer }")

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > self.epsilon :
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(self.nA)

    def sarsamax_step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        - trying out off policy 
        """
        current = self.Q[state][action]
        Q_sa = np.max(self.Q[next_state])
        self.Q[state][action] = current + self.alpha * ((reward + (self.gamma * Q_sa)) - current)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        - expected sarsa off policy
        """
        current = self.Q[state][action]
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        policy_s[np.argmax(self.Q[next_state])] = 1 - self.epsilon + (self.epsilon/self.nA)
        Q_sa = np.dot(self.Q[next_state], policy_s)
        self.Q[state][action] = current + self.alpha * ((reward + (self.gamma * Q_sa)) - current)
        if done: 
            self.epsilon /= self.epsilonreducer