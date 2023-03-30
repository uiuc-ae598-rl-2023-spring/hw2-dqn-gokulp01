import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.activation = nn.Tanh()
        
    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

import random
from collections import namedtuple

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):

        e = Experience(state, action, reward, next_state, done)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = e
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):

        return random.sample(self.memory, batch_size)
    
    def __len__(self):

        return len(self.memory)


class DQN:
    def __init__(self, state_dim, action_dim, memory_capacity=10000, batch_size=64, gamma=0.95, learning_rate=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.argmax(1).item()
        
    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        states = torch.tensor(batch.state, dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(batch.next_state, dtype=torch.float32).to(self.device)
        dones = torch.tensor(batch.done, dtype=torch.uint8).unsqueeze(1).to(self.device)
        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)
        
    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

