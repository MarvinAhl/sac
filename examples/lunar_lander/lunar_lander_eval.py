import sys
sys.path.insert(1, '../..')

from sac import SAC

import torch

import gym

import numpy as np
from matplotlib import pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')

env = gym.make('LunarLanderContinuous-v2')

nS = env.observation_space.shape[0]
nA = env.action_space.shape[0]

agent = SAC(nS, nA, policy_hidden=(128, 128), value_hidden=(128, 128), buffer_size_max=30000,
             buffer_size_min=256, device=device)
agent.load_net('lunar_lander.net')

while True:
    obsv, done = env.reset(), False

    while not done:
        env.render()

        actions = agent.act_deterministic(obsv)
        new_obsv, _, done, _ = env.step(actions)

        obsv = new_obsv