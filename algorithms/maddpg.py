import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from gridworld_env import GridworldEnv

# class ActorNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, action_dim)
#         self.tanh = nn.Tanh()

#     def forward(self, state):
#         x = self.fc1(state)
#         x = self.tanh(x)
#         x = self.fc2(x)
#         x = self.tanh(x)
#         x = self.fc3(x)
#         x = self.tanh(x)
#         return x

# class ActorNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_dim // 2, 64)  # Divide state_dim by 2 to get the state size for each agent
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, action_dim)
#         self.tanh = nn.Tanh()

#     def forward(self, state):
#         # Assuming the state tensor has the shape (batch_size, state_dim)
#         # Split the state tensor into two parts, one for each agent
#         state_1 = state[:, :state_dim // 2]
#         state_2 = state[:, state_dim // 2:]

#         x1 = self.fc1(state_1)
#         x1 = self.tanh(x1)
#         x1 = self.fc2(x1)
#         x1 = self.tanh(x1)
#         x1 = self.fc3(x1)
#         x1 = self.tanh(x1)

#         x2 = self.fc1(state_2)
#         x2 = self.tanh(x2)
#         x2 = self.fc2(x2)
#         x2 = self.tanh(x2)
#         x2 = self.fc3(x2)
#         x2 = self.tanh(x2)

#         return torch.cat([x1, x2], dim=1)  # Concatenate the actions for both agents

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim // 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.fc1(state)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        return x

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        # Ensure action tensor has a single dimension
        action = action.view(-1, action_dim)  # Reshape action to (batch_size, action_dim)

        x = torch.cat([state, action], dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

# class CriticNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(CriticNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_dim + action_dim, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, 1)

#     def forward(self, state, action):
#         x = torch.cat([state, action], dim=1)
#         x = self.fc1(x)
#         x = torch.relu(x)
#         x = self.fc2(x)
#         x = torch.relu(x)
#         x = self.fc3(x)
#         return x

# class CriticNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(CriticNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_dim + action_dim, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, 1)

#     def forward(self, state, action):
#         x = torch.cat([state, action], dim=1)
#         k = torch.cat([state, action], dim=1)
#         print(k.shape)  
#         x = self.fc1(x)
#         x = torch.relu(x)
#         x = self.fc2(x)
#         x = torch.relu(x)
#         x = self.fc3(x)
#         return x



class MADDPGAgent:
    def __init__(self, env, state_dim, action_dim, agent_id, lr_actor, lr_critic, gamma, tau):
        self.agent_id = agent_id
        self.env = env
        self.actor = ActorNetwork(state_dim // 2, action_dim)  # Adjust state_dim for individual agent
        self.critic = CriticNetwork(state_dim, action_dim)
        self.actor_target = ActorNetwork(state_dim // 2, action_dim)  # Adjust state_dim for individual agent
        self.critic_target = CriticNetwork(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.tau = tau

        # Initialize target networks
        self.soft_update(self.actor, self.actor_target, 1.0)
        self.soft_update(self.critic, self.critic_target, 1.0)

    def act(self, state, epsilon=0.0):
        if np.random.rand() < epsilon:
            return self.env.action_space[self.agent_id].sample()
        else:
            state = torch.FloatTensor(state)
            state = state[:, self.agent_id * (state_dim // 2):(self.agent_id + 1) * (state_dim // 2)]  # Extract agent's own state
            action = self.actor(state)
            return action.detach().numpy()

    def update(self, batch, other_agents):
        state = torch.FloatTensor(batch['state'])
        action = torch.FloatTensor(batch['action'])
        reward = torch.FloatTensor(batch['reward'])
        next_state = torch.FloatTensor(batch['next_state'])
        done = torch.FloatTensor(batch['done'])

        # Update critic network
        next_actions = torch.cat([agent.actor_target(next_state) for agent in other_agents], dim=1)
        target_q = self.critic_target(next_state, next_actions)
        expected_q = reward + self.gamma * (1 - done) * target_q
        critic_loss = nn.MSELoss()(self.critic(state, action), expected_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor network
        my_action = self.actor(state[:, self.agent_id * (state_dim // 2):(self.agent_id + 1) * (state_dim // 2)])
        other_actions = torch.cat([agent.actor(state[:, i * (state_dim // 2):(i + 1) * (state_dim // 2)]) for i, agent in enumerate(other_agents) if i != self.agent_id], dim=1)
        actor_loss = -self.critic(state[:, self.agent_id * (state_dim // 2):(self.agent_id + 1) * (state_dim // 2)], torch.cat([my_action, other_actions], dim=1))
        actor_loss = actor_loss.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.actor_target, self.tau)
        self.soft_update(self.critic, self.critic_target, self.tau)

    def soft_update(self, source, target, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class MADDPGTrainer:
    def __init__(self, env, state_dim, action_dim, lr_actor, lr_critic, gamma, tau, buffer_size, batch_size):
        self.env = env
        self.agents = [MADDPGAgent(env, state_dim, action_dim, i, lr_actor, lr_critic, gamma, tau) for i in range(2)]
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

    def train(self, num_episodes, max_steps, epsilon_start, epsilon_end, epsilon_decay):
        for episode in range(num_episodes):
            state = self.env.reset()
            epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
            done = [False] * 2

            for step in range(max_steps):
                actions = [agent.act(state[i], epsilon) for i, agent in enumerate(self.agents)]
                next_state, rewards, done = self.env.step(actions)
                self.replay_buffer.add(state, actions, rewards, next_state, done)

                if all(done):
                    break

                batch = self.replay_buffer.sample(self.batch_size)
                for agent in self.agents:
                    agent.update(batch, [self.agents[i] for i in range(2) if i != agent.agent_id])

                state = next_state

            self.env.render(sum(rewards), done[0], f'episode_{episode}.png')

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'done': []}

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer['state']) >= self.buffer_size:
            self.buffer['state'].pop(0)
            self.buffer['action'].pop(0)
            self.buffer['reward'].pop(0)
            self.buffer['next_state'].pop(0)
            self.buffer['done'].pop(0)

        self.buffer['state'].append(state)
        self.buffer['action'].append(action)
        self.buffer['reward'].append(reward)
        self.buffer['next_state'].append(next_state)
        self.buffer['done'].append(done)

    # def sample(self, batch_size):
    #     indices = np.random.choice(len(self.buffer['state']), size=batch_size, replace=False)
    #     batch = {key: [self.buffer[key][i] for i in indices] for key in self.buffer}
    #     return batch
    def sample(self, batch_size):
        if len(self.buffer['state']) < batch_size:
            indices = np.random.choice(len(self.buffer['state']), size=batch_size, replace=True)
        else:
            indices = np.random.choice(len(self.buffer['state']), size=batch_size, replace=False)
        batch = {key: [self.buffer[key][i] for i in indices] for key in self.buffer}
        return batch

if __name__ == "__main__":
    env = GridworldEnv('6', from_file=True)
    state_dim = 64
    action_dim = 5
    num_agents = 2
    lr_actor = 1e-4
    lr_critic = 1e-3
    gamma = 0.99
    tau = 0.01
    buffer_size = 10000
    batch_size = 64
    num_episodes = 1000
    max_steps = 300
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995

    trainer = MADDPGTrainer(env, state_dim, action_dim, num_agents, lr_actor, lr_critic, gamma, tau, buffer_size, batch_size)
    trainer.train(num_episodes, max_steps, epsilon_start, epsilon_end, epsilon_decay)