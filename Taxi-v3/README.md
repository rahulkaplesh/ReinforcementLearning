** Estimating the Reinforcement learning parameters for taxi-v3 agent **

# Aim 

This was a mini project taken up during the udacity deep reinforcement learning nanodegree . Aim was to solve the taxi-v3 agent

---

# Tools used 

1.) pyeasyga - genetic algorithm for refining the hyperparameters 
2.) python

---

# Method Implemented 

I implemented the expected sarsa learning method . To tune the hyperparameters i used Genetic Algorithms and refgined the 
tuning ranges . I found the best results in the range :

data = { 'epsilon' : [0.09, 0.19], 'gamma' : [0.5, 0.6], 'epsilonreducer' : [85, 100] }

I managed to attain a reward of 8.97 in my last generations . Though in these parameters rewards are generally above 8 ..

Attached the jupyter notebook in which i had writted my test code