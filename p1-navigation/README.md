# Project 1: Navigation

### Introduction

For this project, an agent is trained to collect bananas in a square world !!

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting started

1. You can start by cloning this repo as i have downloaded and included the unity versions of the environment which i used to do it as well as the code works for the directory structure

### Instruction

The excercise was given as two challenges 

1.) One version of the enevironment gave a vector state space using which the machine learning agent could be trained 

2.) Another version of the environment gave a RGB image as the state space which had to be run through a cnn to decipher the state space and take actions accordingly

### Directory Structure

Banana_Windows_x86_x64 : Directory containing the unity environment which gave the vector state space to work with.
DoubleDQN : Implemetation of Double DQN to solve the environment which returns the vector state space .64 x 64 layer fc network was used as deep nn to estimate the action values.
DobubleDQN_moreFCLayers : Implemetation of Double DQN to solve the environment which returns the vector state space .128 x 64 x 32 layer fc network was used as deep nn to estimate the action values.
DoubleDQN4LayersExpReplay : Implementation of Double DQN with prioritised replay to solve the environment which returns the vector state space. 128 x 64 x 32 layer fc deep nn was used to estimate the action values.
Dueling_NN_DQN : Implementation of Double DQN with duelling nueral network for estimating the value of the state. Both the networks were 128 x 64 x 32 layer fc nn to extimate the action values.
DQN_RNN_PixelState : Implementation of Double DQN on the environment which returned the RGB image of what is seen by the agent . A set of 4 such images were stitched together to form a frame which was to run through a CNN . Any new state was added on to the end of the stack and the image from the top of the stack was removed. This frame was then run through a 3D CNN to extract the state 
features out of the frame of 4 state captures. These features were used to determine the actions which the agent can take within those states

### Result files 

This repository also consists of a Report.pdf file which contain some of my observations.
