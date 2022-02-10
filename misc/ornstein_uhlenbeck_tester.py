"""
Little script to test different alpha and beta parameters
for Normal Ornstein-Uhlenbeck-Process (with tanh squeezing) for DRL exploration.

How to tune:
1. Set steps to your desired lookahead horizon.
2. Set alpha such that the time correlation is as strong as you need it.
3. Then change beta until the RMS (red dashed line) is at about 0.7 to 0.75.
If it doesn't look right, repeat steps 2 and 3 until it does.
"""

import torch
from matplotlib import pyplot as plt

steps = 50
a = 0.3  # alpha - squeezes horizontally and squeezes vertically
b = 0.8  # beta - stretches vertically

samples = 4
O = torch.zeros((samples, steps))

for i in range(steps-1):
    N = torch.normal(0.0, 1.0, size=(samples,))
    O[:, i+1] = O[:, i] - a*O[:, i] + b*N

O = torch.tanh(O)
rms_O = O.pow(2).mean(axis=-1, keepdims=True).pow(0.5).expand_as(O)

# Show 4 example plots
fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(torch.arange(len(O[0])), O[0], 'k-')
ax[0, 0].plot(torch.arange(len(rms_O[0])), rms_O[0], 'r--')
ax[0, 0].set_ylim((-1.0, 1.0))
ax[0, 0].grid(True)
ax[0, 1].plot(torch.arange(len(O[1])), O[1], 'k-')
ax[0, 1].plot(torch.arange(len(rms_O[1])), rms_O[1], 'r--')
ax[0, 1].set_ylim((-1.0, 1.0))
ax[0, 1].grid(True)
ax[1, 0].plot(torch.arange(len(O[2])), O[2], 'k-')
ax[1, 0].plot(torch.arange(len(rms_O[2])), rms_O[2], 'r--')
ax[1, 0].set_ylim((-1.0, 1.0))
ax[1, 0].grid(True)
ax[1, 1].plot(torch.arange(len(O[3])), O[3], 'k-')
ax[1, 1].plot(torch.arange(len(rms_O[3])), rms_O[3], 'r--')
ax[1, 1].set_ylim((-1.0, 1.0))
ax[1, 1].grid(True)

plt.savefig('ornstein_uhlenbeck.png', dpi=500)