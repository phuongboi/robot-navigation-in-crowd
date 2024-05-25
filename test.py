import numpy as np
from collections import deque
import random
import torch
from torch import nn
import os
from env import CrowdEnv
from matplotlib import pyplot as plt
import re

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
    def __init__(self, model_path, env):

        self.env = env
        self.model_path = model_path
        self.num_actions = 80
        self.robot_dim = 6
        self.human_dim = 7
        self.lstm_hidden_dim = 48
        self.model = self.make_model()
        self.model.load_state_dict(torch.load(self.model_path))

    def make_model(self):
        model = Network(self.robot_dim, self.human_dim, self.lstm_hidden_dim, self.num_actions)
        return model

    def agent_policy(self, state):
        q_value = self.model(state)
        action = np.argmax(q_value.detach().numpy())
        #print(action)
        return action


    def test(self, num_episodes=500, test_case=None):
        self.model.eval()
        rewards_list = []
        success_times_list = []
        goal_count = 0
        collide_count = 0
        timeout_count = 0
        for episode in range(num_episodes):
            if num_episodes == 1 and test_case != None:
                obs = env.reset(human_num=5, test_phase=True, counter=test_case)
            else:
                obs = env.reset(human_num=5, test_phase=True, counter=episode)
            state = env.convert_coord(obs)
            reward_for_episode = 0
            while True:
                if num_episodes == 1 and test_case != None:
                    env.render()
                received_action = self.agent_policy(state)
                vel_action = env.vel_samples[received_action]
                next_obs, reward, terminal, info = env.step(vel_action)
                next_state = env.convert_coord(next_obs)
                # add up rewards
                reward_for_episode += reward
                state = next_state
                if terminal:
                    rewards_list.append(reward_for_episode)
                    if info == "timeout":
                        timeout_count += 1
                    elif info == "collide":
                        collide_count += 1
                    elif "Goal" in info:
                        goal_count += 1
                        success_times_list.append(float(re.findall("\d+\.\d+", info)[0]))
                    else:
                        raise ValueError("Info format")

                    print("Episode: {} done, Reward: {}, Status: {}".format(episode, reward_for_episode, info))
                    break
        print(timeout_count + collide_count + goal_count)
        assert (timeout_count + collide_count + goal_count) == num_episodes
        print("Evaluate on {} test case: Success rate: {}, Collide rate: {}, Timeout rate: {}".format \
                (num_episodes, goal_count/(num_episodes), collide_count/(num_episodes), timeout_count/(num_episodes)))
        print("Average reward: {}, Average success nav time: {}".format(np.mean(rewards_list),np.mean(success_times_list)))

if __name__ == "__main__":
    env = CrowdEnv()
    num_episodes = 10
    model_path = "weights2/ep_3729.pth"
    model = DQN(model_path, env)
    model.test(num_episodes=num_episodes, test_case=100)
