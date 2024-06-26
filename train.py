import numpy as np
from collections import deque
import random
import torch
from torch import nn
import os
from env import CrowdEnv
from matplotlib import pyplot as plt
from IPython.display import clear_output

class Network(nn.Module):
    def __init__(self,robot_dim, human_dim, lstm_hidden_dim, num_actions):
        super().__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.robot_dim = robot_dim

        self.lstm = nn.LSTM(human_dim, lstm_hidden_dim, batch_first =True)
        self.mlp = nn.Sequential(
            nn.Linear(robot_dim + lstm_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, state):
        size = state.shape
        robot_state = state[:, 0, :self.robot_dim]
        human_state = state[:, :, self.robot_dim:]

        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(human_state, (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([robot_state, hn], dim=1)
        q_value = self.mlp(joint_state)

        return q_value

class DQN:
    def __init__(self, model_path, env, lr, batch_size, gamma, eps_decay, eps_start, eps_end, initial_memory, memory_size):

        self.env = env
        self.model_path = model_path
        self.lr = lr
        self.gamma = gamma
        self.eps_decay = eps_decay
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.initial_memory = initial_memory

        self.replay_buffer = deque(maxlen=memory_size)
        self.replay_buffer_b = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.num_actions = 80
        self.robot_dim = 6
        self.human_dim = 7
        self.lstm_hidden_dim = 48
        self.model = self.make_model()
        self.first = False # for debug

    def make_model(self):
        model = Network(self.robot_dim, self.human_dim, self.lstm_hidden_dim, self.num_actions)
        return model

    def agent_policy(self, state, epsilon):
        # epsilon greedy policy
        if np.random.rand() < epsilon:
            action = random.randrange(self.num_actions)
        else:
            q_value = self.model(state)
            action = np.argmax(q_value.detach().numpy())

        return action


    def add_to_replay_buffer(self, state, action, reward, next_state, terminal):
        self.replay_buffer.append((state, action, reward, next_state, terminal))

    def sample_from_reply_buffer(self):
        random_sample = random.sample(self.replay_buffer, self.batch_size)
        random_sample_b = random.sample(self.replay_buffer_b, int(self.batch_size/2))
        return random_sample + random_sample_b

    def get_memory(self, random_sample):
        states = torch.cat([i[0] for i in random_sample], dim=0)
        actions = torch.tensor([i[1] for i in random_sample])
        rewards = torch.tensor([i[2] for i in random_sample])
        next_states = torch.cat([i[3] for i in random_sample], dim=0)
        terminals = torch.tensor([i[4] for i in random_sample]) * 1

        return states, actions, rewards, next_states, terminals

    def train_with_relay_buffer(self):
        # replay_memory_buffer size check
        if len(self.replay_buffer) < self.batch_size:
            return

        sample = self.sample_from_reply_buffer()
        states, actions, rewards, next_states, terminals = self.get_memory(sample)
        next_q_mat = self.model(next_states)

        next_q_vec = np.max(next_q_mat.detach().numpy(), axis=1).squeeze()

        target_vec = rewards + self.gamma * next_q_vec * (1 - terminals.detach().numpy())
        q_mat = self.model(states)
        q_vec = q_mat.gather(dim=1, index=actions.unsqueeze(1)).type(torch.FloatTensor)
        target_vec = target_vec.unsqueeze(1).type(torch.FloatTensor)
        loss = self.loss_func(q_vec, target_vec)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self, num_episodes=2000):
        self.model.train()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        steps_done = 0
        losses = []
        rewards_list = []
        for episode in range(num_episodes):
            obs = env.reset(human_num=5, test_phase=False, counter=None)
            state = env.convert_coord(obs)
            reward_for_episode = 0
            num_step_per_eps = 0
            ep_buffer = []
            while True:
                epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(- steps_done / self.eps_decay)
                received_action = self.agent_policy(state, epsilon)
                vel_action = env.vel_samples[received_action]
                steps_done += 1
                num_step_per_eps += 1
                next_obs, reward, terminal, info = env.step(vel_action)
                next_state = env.convert_coord(next_obs)
                # Store the experience in replay memory
                #self.add_to_replay_buffer(state, received_action, reward, next_state, terminal)
                ep_buffer.append((state, received_action, reward, next_state, terminal))
                # add up rewards
                reward_for_episode += reward
                state = next_state

                if max(len(self.replay_buffer), len(self.replay_buffer_b)) >= self.initial_memory and steps_done % 4 == 0:
                    loss = self.train_with_relay_buffer()
                    losses.append(loss.item())

                if steps_done % 10000 == 0:
                    plot_stats(steps_done, rewards_list, losses, episode)
                    path = os.path.join(self.model_path, f"ep_{episode}.pth")
                    torch.save(self.model.state_dict(), path)

                if max(len(self.replay_buffer), len(self.replay_buffer_b)) >= self.initial_memory and not self.first:
                    print("Start learning from buffer")
                    print(len(self.replay_buffer))
                    print(len(self.replay_buffer_b))
                    self.first = True
                if terminal:
                    rewards_list.append(reward_for_episode)
                    if info == "collide":
                        self.replay_buffer_b += ep_buffer
                    else:
                        self.replay_buffer += ep_buffer
                    print("Episode: {} done, Reward: {}, Status: {}".format(episode, reward_for_episode, info))
                    break



def plot_stats(frame_idx, rewards, losses, step):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title(f'Total frames {frame_idx}. Avg reward over last 10 episodes: {np.mean(rewards[-10:])}')
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    #plt.show()
    plt.savefig('figures2/fig_{}.png'.format(step))


if __name__ == "__main__":
    env = CrowdEnv()

    # setting up params
    lr = 0.0001
    batch_size = 64
    eps_decay = 50000
    eps_start = 0.5
    eps_end = 0.1
    initial_memory = 10000
    memory_size = 10 * initial_memory
    gamma = 0.9
    num_episodes = 10000
    model_path = "weights2/"
    print('Start training')
    model = DQN(model_path, env, lr, batch_size, gamma, eps_decay, eps_start, eps_end,initial_memory, memory_size)
    model.train(num_episodes)
