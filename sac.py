"""
Author: Marvin Ahlborn

This is an implementation of the Soft Actor-Critic (SAC)
Algorithm using Pytorch and quite a bit of entropy.
"""

import torch
from torch import tensor
from torch import nn
import numpy as np

class PolicyNetwork(nn.Module):
    """
    The deterministic Policy Network
    """
    def __init__(self, state, actions, hidden=(512, 512)):
        """
        state: Integer telling the state dimension
        actions: Integer telling the number of actions
        hidden: Tuple of the each hidden layers nodes
        """
        super(PolicyNetwork, self).__init__()

        modules = []

        modules.append(nn.Linear(state, hidden[0]))
        modules.append(nn.LeakyReLU(0.1))

        for i in range(len(hidden) - 1):
            modules.append(nn.Linear(hidden[i], hidden[i+1]))
            modules.append(nn.LeakyReLU(0.1))

        self.module_stack = nn.Sequential(*modules)
        
        self.mu_layer = nn.Linear(hidden[-1], actions)  # Mean of stoachstic policy value
        self.log_std_layer = nn.Linear(hidden[-1], actions)  # Logarithm of standard deviation
    
    def forward(self, state, stochastic=True):
        module_output = self.module_stack(state)
        mu = self.mu_layer(module_output)
        log_std = self.log_std_layer(module_output)
        log_std_clip = torch.clip(log_std, -20.0, 2.0)  # Have it in reasonable bounds
        std = torch.exp(log_std_clip)

        if stochastic:
            pi_dist = torch.distributions.normal.Normal(mu, std)
            unbound_actions = pi_dist.rsample()
            actions = torch.tanh(unbound_actions)

            # Second term is result of applying tanh to the actions
            log_probs = pi_dist.log_prob(unbound_actions).sum(dim=-1, keepdim=True) - \
                torch.log(1.0 - actions**2).sum(dim=-1, keepdim=True)
        else:
            actions = torch.clip(mu, -1.0, 1.0)
            log_probs = None

        return actions, log_probs

class ValueNetwork(nn.Module):
    """
    Twin Q-Network
    """
    def __init__(self, state, actions, hidden=(512, 512)):
        """
        state: Integer telling the state dimension
        actions: Integer telling the number of actions
        hidden: Tuple of the each hidden layers nodes
        """
        super(ValueNetwork, self).__init__()

        modules_a = []
        modules_b = []

        modules_a.append(nn.Linear(state + actions, hidden[0]))  # Takes in State and Action
        modules_b.append(nn.Linear(state + actions, hidden[0]))
        modules_a.append(nn.LeakyReLU(0.1))
        modules_b.append(nn.LeakyReLU(0.1))

        for i in range(len(hidden) - 1):
            modules_a.append(nn.Linear(hidden[i], hidden[i+1]))
            modules_b.append(nn.Linear(hidden[i], hidden[i+1]))
            modules_a.append(nn.LeakyReLU(0.1))
            modules_b.append(nn.LeakyReLU(0.1))
        
        modules_a.append(nn.Linear(hidden[-1], 1))  # Only one output for the State-Action-Value
        modules_b.append(nn.Linear(hidden[-1], 1))

        self.module_stack_a = nn.Sequential(*modules_a)
        self.module_stack_b = nn.Sequential(*modules_b)
    
    def forward(self, state, actions):
        state_action_input = torch.cat((state, actions), dim=-1)
        value_output_a = self.module_stack_a(state_action_input)
        value_output_b = self.module_stack_b(state_action_input)
        return value_output_a, value_output_b

class ReplayBuffer:
    """
    Uniformly random Replay Buffer
    """
    def __init__(self, state, actions, max_len=50000):
        """
        state: Integer of State Dimension
        actions: Integer of Number of Actions
        """
        self.states = np.empty((max_len, state), dtype=np.float32)
        self.actions = np.empty((max_len, actions), dtype=np.float32)
        self.rewards = np.empty((max_len, 1), dtype=np.float32)
        self.next_states = np.empty((max_len, state), dtype=np.float32)
        self.terminals = np.empty((max_len, 1), dtype=np.int8)

        self.index = 0
        self.full = False
        self.max_len = max_len
        self.rng = np.random.default_rng()
    
    def store_experience(self, state, actions, reward, next_state, terminal):
        """
        Stores given SARS Experience in the Replay Buffer.
        Returns True if the last element has been written to memory and the
        Buffer will start over replacing the first elements at next call.
        """
        self.states[self.index] = state
        self.actions[self.index] = actions
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.terminals[self.index] = terminal
        
        self.index += 1
        self.index %= self.max_len  # Replace oldest Experiences if Buffer is full

        if self.index == 0:
            self.full = True
            return True
        return False

    def get_experiences(self, batch_size):
        """
        Returns batch of experiences for replay.
        """
        indices = self.rng.choice(self.__len__(), batch_size)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        terminals = self.terminals[indices]

        return states, actions, rewards, next_states, terminals

    def __len__(self):
        return self.max_len if self.full else self.index

class SAC:
    def __init__(self, state, actions, policy_hidden=(512, 512), value_hidden=(512, 512), gamma=0.99,
                 learning_rate=0.0002, learning_rate_alpha=0.001, alpha=1.0, buffer_size_max=50000,
                 buffer_size_min=1024, batch_size=64, replays=1, tau=0.01, device='cpu'):
        """
        state: Integer of State Dimension
        actions: Integer of Number of Action
        """
        self.state = state
        self.actions = actions

        self.policy_hidden = policy_hidden
        self.policy_net = PolicyNetwork(state, actions, policy_hidden).to(device)
        
        self.value_hidden = value_hidden
        self.value_net = ValueNetwork(state, actions, value_hidden).to(device)
        self.target_value_net = ValueNetwork(state, actions, value_hidden).to(device)

        self._update_targets(1.0)  # Fully copy Online Net weights to Target Net
  
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)

        # The entropy temperature alpha, it is automatically tuned
        self.log_alpha = tensor([np.log(alpha)], dtype=torch.float32, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate_alpha)

        self.alpha_start = alpha

        self.target_entropy = -actions  # Used to update alpha

        self.learning_rate = learning_rate
        self.learning_rate_alpha = learning_rate_alpha

        self.buffer = ReplayBuffer(state, actions, buffer_size_max)
        self.buffer_size_max = buffer_size_max
        self.buffer_size_min = buffer_size_min
        self.batch_size = batch_size
        self.replays = replays  # On how many batches it should train after each step

        # Can be calculated by exp(- dt / lookahead_horizon)
        self.gamma = gamma  # Reward discount rate

        self.rng = np.random.default_rng()

        self.tau = tau  # Mixing parameter for polyak averaging of target and online network

        self.device = device
    
    def reset(self):
        """
        Reset object to its initial state if you want to do multiple training passes with it
        """
        self.policy_net = PolicyNetwork(self.state, self.actions, self.policy_hidden).to(self.device)
        self.value_net = ValueNetwork(self.state, self.actions, self.value_hidden).to(self.device)
        self.target_value_net = ValueNetwork(self.state, self.actions, self.value_hidden).to(self.device)
        self._update_targets(1.0)  # Fully copy Online Net weights to Target Net

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)

        self.log_alpha = tensor([np.log(self.alpha_start)], dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate_alpha)

        self.buffer = ReplayBuffer(self.state, self.actions, self.buffer_size_max)

        self.rng = np.random.default_rng()

    def act(self, state):
        """
        Decides on action based on current state using Policy Net.
        """
        # TODO: Add Warmup episodes
        with torch.no_grad():
            state = tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
            actions, _ = self.policy_net(state)
            actions = actions.squeeze().cpu().numpy()

        return actions
    
    def act_deterministic(self, state):
        """
        Decides on action but instead of sampling from normal distribution uses mean values directly.
        """
        with torch.no_grad():
            state = tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
            actions, _ = self.policy_net(state, stochastic=False)
            actions = actions.squeeze().cpu().numpy()

        return actions
    
    def experience(self, state, actions, reward, next_state, terminal):
        """
        Takes experience and stores it for replay.
        """
        self.buffer.store_experience(state, actions, reward, next_state, terminal)
    
    def train(self):
        """
        Train Value and Target Networks on batches from replay buffer.
        """
        if len(self.buffer) < self.buffer_size_min:
            return  # Dont train until Replay Buffer has collected a certain number of initial experiences

        for _ in range(self.replays):
            states, actions, rewards, next_states, terminals = self.buffer.get_experiences(self.batch_size)

            states = torch.from_numpy(states).to(self.device)
            rewards = torch.from_numpy(rewards).to(self.device)
            actions = torch.from_numpy(actions).to(self.device)
            next_states = torch.from_numpy(next_states).to(self.device)
            terminals = torch.from_numpy(terminals).to(self.device)

            alpha = torch.exp(self.log_alpha).item()

            # Q-Function training
            next_actions, next_log_probs = self.policy_net(next_states)
            next_values_a, next_values_b = self.target_value_net(next_states, next_actions)

            td_targets = rewards + self.gamma * (1 - terminals) * \
                (torch.min(next_values_a, next_values_b).detach() - alpha * next_log_probs)
            predictions_a, predictions_b = self.value_net(states, actions)
            td_errors_a = td_targets - predictions_a
            td_errors_b = td_targets - predictions_b
            value_loss = td_errors_a.pow(2).mean() + td_errors_b.pow(2).mean()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # Policy Function training
            pred_actions, log_probs = self.policy_net(states)
            pred_values_a, pred_values_b = self.value_net(states, pred_actions)
            policy_loss = -(torch.min(pred_values_a, pred_values_b) - alpha * log_probs).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Entropy Temperature update
            alpha_loss = -(self.log_alpha * (self.target_entropy + log_probs).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self._update_targets(self.tau)
    
    def save_net(self, path):
        torch.save(self.policy_net.state_dict(), 'policy_' + path)
        torch.save(self.value_net.state_dict(), 'value_' + path)
    
    def load_net(self, path):
        self.policy_net.load_state_dict(torch.load('policy_' + path))
        self.value_net.load_state_dict(torch.load('value_' + path))
        self._update_targets(1.0)  # Also load weights into target net
    
    def _update_targets(self, tau):
        """
        Update Target Networks by blending Target und Online Network weights using the factor tau (Polyak Averaging)
        A tau of 1 just copies the whole online network over to the target network
        """
        for value_param, target_value_param in zip(self.value_net.parameters(), self.target_value_net.parameters()):
            target_value_param.data.copy_(tau * value_param.data + (1.0 - tau) * target_value_param.data)