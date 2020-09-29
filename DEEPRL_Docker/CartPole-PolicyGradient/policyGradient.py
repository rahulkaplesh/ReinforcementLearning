import torch.optim as optim
import torch
import gym
from agent import Policy
from collections import deque
import numpy as np

env = gym.make('CartPole-v1')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

n_episodes = 2000
max_t = 1000
gamma = 1.0
print_every = 100

scores_deque = deque(maxlen=100)
scores = []

for i_episode in range(1, n_episodes+1):
    saved_log_probs = []
    rewards = []
    state = env.reset()
    for t in range(max_t):
        action, log_prob = policy.act(state)
        saved_log_probs.append(log_prob)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break
    scores_deque.append(sum(rewards))
    scores.append(sum(rewards))

    discounts = [gamma**i for i in range(len(rewards)+1)]
    R = sum([a*b for a,b in zip(discounts, rewards)])

    policy_loss = []
    for log_prob in saved_log_probs:
        policy_loss.append(-log_prob*R)
    policy_loss = torch.cat(policy_loss).sum()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    if i_episode % print_every == 0:
        print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    if np.mean(scores_deque)>=195.0:
        print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
        break