import torch
from torch import tensor
from sac import SAC

agent = SAC(2, 2, policy_hidden=(2, 2), value_hidden=(2, 2),
    buffer_size_max=20, buffer_size_min=0, batch_size=2)

agent.reset()

actions = agent.act([1, 2])

actions = agent.act_deterministic([1, 2])

agent.experience([1, 1], [0, 2], 2, [3, 1], 0)

agent.train()

pass