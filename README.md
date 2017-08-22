# Reinforcement-Learning

# Overview
This repository contains my ongoing implementations of reinforcement learning algorithms. At present, the algorithms are fully compatible to be run with  environments from OpenAI Gym that have discrete state and action spaces, and a Grid World environment I created to test algorithms with. In addition to this, I have also created a testbed for multi-arm bandit problems to allow for comparison of regret between algorithms. The bulk of the code is contained within RL.py, example uses with OpenAI Gym are in code/OpenAIGYMExamples.ipynb, example uses with Grid World are contained in code/GridWorld.ipynb, and example uses with Bandits are contained in code/Bandits.ipynb. The figs/ folder contains example Grid World q-value, value, and policy visualizations, as well as visualizations of episode returns and hyperparameter choices.

# Visualizations


# Model Based Reinforcement Learning
Current list of model based reinforcement learning (dynamic programming) algorithms:
- Iterative Policy Evaluation
- Policy Iteration
- Value Iteration
- Q-Value Iteration

# Reinforcement Learning Policy Strategies
The model free reinforcement learning algorithms have the option of several exploration-exploitation strategies. 
- Greedy
- Epsilon Greedy
- Epsilon Greedy with Exponential Epsilon Decay (Exponential Increase in Greediness)
- Boltzmann 
- Boltzmann with Exponential Temperature Decay (Exponential Increase in Greediness)

# Learning Models
I have implemented a class to learn a model (Transition and Reward Kernels) by simulating randomly in an environment for some amount of time. This is useful if a model is not given a-priori and you are interested in using model based reinforcement learning algorithms.

# Model Free Reinforcement Learning
Current list of model free reinforcement learning algorithms:
- One Step Temporal Differences
- Sarsa (On Policy Temporal Difference Learning)
- Q-Learning (Off Policy Temporal Difference Learning)

# Risk-Sensitive Reinforcement Learning
Current list of risk-sensitive reinforcement learning algorithms:
- Expected Utility Q-Learning (Nonlinear Mapping of Rewards with Value Function)
- Risk-Sensitive Q-Learning (Nonlinear Mapping of Temporal Differences with Value Function)


Current list of risk-sensitive value functions:
- Prospect Theory Value Function
- Logarithmic Value Function
- Entropic Value Function

# Multi-Arm Bandits
Current list of multi-arm bandit algorithms:
- Greedy
- Epsilon Greedy
- Upper Confidence Bounds (UCB)