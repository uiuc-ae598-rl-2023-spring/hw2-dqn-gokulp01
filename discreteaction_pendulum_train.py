import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import numpy as np
import random
import math
import matplotlib
import matplotlib.pyplot as plt
import discreteaction_pendulum
import pandas as pd
import os 
# from config import reset_hyp
from hyperparameters import *
import Plotter


print(num_episodes)
env = discreteaction_pendulum.Pendulum()
n_observations = env.num_states
n_actions = env.num_actions
modes=[(10000, 500), (10000, 1), (128, 500), (128, 1)]


# DQN Class with 2 hidden layers and tanh activation function
class DQN(nn.Module):
    def __init__(self, n_observation, n_actions, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(n_observation, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)
    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        return self.layer3(x)

# Transition tuple
class Transition(namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))):
    pass

# Replay Memory Class with pushing functionality
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)   
    def push(self, state, action, next_state, reward):
        self.memory.append(Transition(state, action, next_state, reward))     
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) 
    def __len__(self):
        return len(self.memory)

# Agent Class 
class Agent:
    def __init__(self, n_observations, n_actions, hidden_size, learning_rate, gamma,
                 epsilon_start, epsilon_end, epsilon_decay, tau, device):
        self.n_observations, self.n_actions, self.hidden_size = n_observations, n_actions, hidden_size
        self.learning_rate, self.gamma, self.tau  = learning_rate, gamma, tau
        self.epsilon_start, self.epsilon_end, self.epsilon_decay  = epsilon_start, epsilon_end, epsilon_decay
        self.anneal_steps, self.device = 1e6, device
        self.policy_network = DQN(n_observations, n_actions, hidden_size).to(device)
        self.target_network = DQN(n_observations, n_actions, hidden_size).to(device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.AdamW(self.policy_network.parameters(), lr=learning_rate, amsgrad=True)
        self.init_training_params()
# Training Parameters
    def init_training_params(self):
        self.memory_size = 10000
        self.memory = ReplayMemory(self.memory_size)
        self.batch_size = 32
        self.update_freq = 1000
        self.steps = 0
# Epsilong Greedy Action
    def choose_action(self, state):
        self.steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon_start - (self.steps / self.anneal_steps) * (self.epsilon_start - self.epsilon_end))
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1 * self.steps / self.epsilon_decay)
        action = np.random.choice(self.n_actions) if np.random.rand() < self.epsilon else self.policy_network(state).argmax(dim=1).item()
        return torch.tensor([[action]], device=self.device, dtype=torch.long)
# Experience Replay 
    def replay_and_learn(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask, non_final_next_states = self.get_non_final_states(batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)
        next_state_values = self.get_next_state_values(non_final_mask, non_final_next_states)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = self.calculate_loss(state_action_values, expected_state_action_values)
        self.update_policy_network(loss)
        self.soft_update_target_network()

    def get_non_final_states(self, next_states):
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)),
                                    device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in next_states if s is not None])
        return non_final_mask, non_final_next_states

    def get_next_state_values(self, non_final_mask, non_final_next_states):
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]
        return next_state_values
# Loss calculation
    def calculate_loss(self, state_action_values, expected_state_action_values):
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        return loss
# policy network update
    def update_policy_network(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()
# target network update
    def soft_update_target_network(self):
        for target_param, policy_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)


def run_episode(agent, env, n_actions, device):
    s = env.reset()
    s = torch.tensor(s, dtype=torch.float32, device=agent.device).unsqueeze(0)
    done = False
    total_reward = 0
    rewards_list = []
    count = 0
    while not done:
        action = agent.choose_action(s) 
        assert(action.item() in np.arange(n_actions))
        obs, r, done = env.step(action.item())
        rewards_list.append(r)
        total_reward += r
        if done:
            s1 = None
        else:
            s1 = torch.tensor(obs, dtype = torch.float32, device = device).unsqueeze(0)
        r = torch.tensor([r], device = agent.device) 
        agent.memory.push(s, action, s1 , r)
        s = s1
        count += 1
        agent.replay_and_learn()
        if agent.steps % agent.update_freq == 0:
            agent.soft_update_target_network()
    return total_reward, rewards_list


def train_agent(agent, env, num_episodes, n_actions, device, weight_dir, reward_dir, reward_file, mode):
    scores = []
    mean_scores = []
    max_mean = 0
    all_udissc_r = []

    for i in range(num_episodes):
        total_reward, rewards_list = run_episode(agent, env, n_actions, device)
        all_udissc_r.append(rewards_list)
        scores.append(total_reward)
        mean = np.mean(scores[-100:])
        mean_scores.append(mean)
        if mean > max_mean:
            torch.save(agent.policy_network.state_dict(), weight_dir+ '/' + mode+ '/'+ 'q_m_wt_mm.pth')
            torch.save(agent.target_network.state_dict(), weight_dir+ '/' + mode+ '/'+'t_m_wt_mm.pth')
            max_mean = mean
        print('Episode {}, Total iterations {}, Total Reward {:.2f}, Mean Reward {:.2f}, Epsilon: {:.2f}'.format(i + 1, agent.steps, total_reward, np.mean(scores[-100:]), agent.epsilon))

    torch.save(agent.policy_network.state_dict(), weight_dir+ '/' + mode+ '/'+ 'q_m_w_e.pth')
    torch.save(agent.target_network.state_dict(), weight_dir+ '/' + mode+ '/'+ 't_m_w_e.pth')
    with open(reward_dir + reward_file, 'w+') as f:
        for items in all_udissc_r:
            f.write('%s\n' %items)
        print("File written successfully")
    f.close()

def run_experiments(modes, n_observations, n_actions, hidden_size, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay, tau, device, batch_size, ann_steps, num_episodes, env, weight_dir, reward_dir, reward_file):
    for mode in range(len(modes)):
        agent = Agent(n_observations, n_actions, hidden_size, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay, tau, device)
        (agent.memory_size, agent.update_freq) = modes[mode]
        agent.batch_size    = batch_size
        agent.anneal_steps  = ann_steps
        train_agent(agent, env, num_episodes, n_actions, device, weight_dir, reward_dir, str(mode) + reward_file, str(mode))


run_experiments(modes, n_observations, n_actions, hidden_size, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay, tau, device, batch_size, ann_steps, num_episodes, env, weight_dir, reward_dir, reward_file)
# to generate gif
plotter.generate_video(fig_dir + 'trajectory_video.gif')