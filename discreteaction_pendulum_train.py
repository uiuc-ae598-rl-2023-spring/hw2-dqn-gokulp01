import random
from collections import namedtuple
import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import discreteaction_pendulum

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

class DQN:
    def __init__(self, state_dim, action_dim, memory_capacity=100000, batch_size=32, gamma=0.95, learning_rate=0.00025):
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

def train_dqn(env_name='Pendulum', num_episodes=1000, max_steps=200, batch_size=32, memory_capacity=100000, gamma=0.95, learning_rate=0.000025, target_update=10, epsilon_decay=0.995, epsilon_min=0.01, render=False):
    env = discreteaction_pendulum.Pendulum()
    state_dim = env.num_states
    action_dim = env.num_actions
    agent = DQN(state_dim, action_dim, memory_capacity, batch_size, gamma, learning_rate)
    epsilon = 1.0
    
    rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            if render:
                env.render()
                
            action = agent.select_action(state, epsilon)
            next_state, reward, done= env.step(action)
            discounted_reward = reward
            
            if not done:
                future_rewards = 0.0
                future_state = next_state
                for future_step in range(step+1, max_steps):
                    future_action = agent.select_action(future_state, 0) 
                    future_state, future_reward, future_done= env.step(future_action)
                    future_rewards += gamma**(future_step-step) * future_reward
                    if future_done:
                        break
                discounted_reward += future_rewards * gamma
            
            episode_reward += discounted_reward
            agent.memory.push(state, action, discounted_reward, next_state, done)
            state = next_state
            
            if done:
                break
                
            if len(agent.memory) > batch_size:
                agent.update_policy()
                
        if episode % target_update == 0:
            agent.update_target()
            
        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        rewards.append(episode_reward)
        print(f"Episode {episode}: Reward = {episode_reward:.2f}")
        agent.save("dqn.pth")
        

    return agent, rewards

agent, rewards = train_dqn(env_name='Pendulum-v0', num_episodes=500, render=False)
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Standard DQN Learning Curve')
# plt.show()
plt.savefig("std_dqn_learning_curve.png")