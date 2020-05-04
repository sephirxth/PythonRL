
import numpy as np
# import matplotlib.pyplot as matplot
# import nni
import random
import replay_buffer
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.layer1 = nn.Linear(4, 16)

        self.layer2 = nn.Linear(16, 32)

        self.layer3 = nn.Linear(32, 2)

        self.optimizer = optim.Adam(self.parameters())
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def predict(self, input):

        return self(input)

    def train(self, input, target_q_value):
        input = torch.Tensor(input)
        target_q_value = torch.Tensor(target_q_value)
        self.optimizer.zero_grad()   # zero the gradient buffers
        current_q_value = self(input)
        loss = self.loss_function(current_q_value, target_q_value)
        loss.backward()
        self.optimizer.step()    # Does the update


class GraphConvolutedQNet(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(GraphConvolutedQNet, self).__init__()
        self.graph_convolution_layer1 = nn.MultiheadAttention(
            embed_dim=1, num_heads=1)

        self.hidden_layer1 = nn.Linear(input_dimension, 16)

        self.hidden_layer2 = nn.Linear(16, 32)

        self.output_layer = nn.Linear(32, output_dimension)

        self.optimizer = optim.Adam(self.parameters())
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        x = torch.Tensor(x)
        x = self.graph_convolution_layer1(x, x, x)
        print("attention return = ", x)
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x

    def predict(self, input):

        return self(input)

    def train(self, input, target_q_value):
        input = torch.Tensor(input)
        target_q_value = torch.Tensor(target_q_value)
        self.optimizer.zero_grad()   # zero the gradient buffers
        current_q_value = self(input)
        loss = self.loss_function(current_q_value, target_q_value)
        loss.backward()
        self.optimizer.step()    # Does the update


class DuelingQNet(nn.Module):
    def __init__(self):
        super(DuelingQNet, self).__init__()
        self.layer1 = nn.Linear(4, 16)

        self.state_value_layer = nn.Linear(16, 1)

        self.action_advantage_layer = nn.Linear(16, 2)

        self.optimizer = optim.Adam(self.parameters())
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.layer1(x))
        state_value = self.state_value_layer(x)
        action_advantage = self.action_advantage_layer(x)
        q_value = state_value+(action_advantage-torch.mean(action_advantage))
        return q_value

    def predict(self, input):

        return self(input)

    def train(self, input, target_q_value):
        input = torch.Tensor(input)
        target_q_value = torch.Tensor(target_q_value)
        self.optimizer.zero_grad()   # zero the gradient buffers
        current_q_value = self(input)
        loss = self.loss_function(current_q_value, target_q_value)
        loss.backward()
        self.optimizer.step()    # Does the update


class DeepQ():
    """

    Parameters
    ----------
        mode: Natural(Basic) Deep Q | Double Deep Q | Dueling Deep Q | 0, 1, 2 三选一

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self,
                 mode=0,
                 env=None,
                 replay_buffer_size=10000,
                 learning_rate=0.01,
                 explore_epsilon=0.5,
                 discount_factor_gamma=0.9,
                 mini_batch_size=32):
        
        self.__action_dimension = env.action_space.n
        self.__state_dimension = env.observation_space.shape[0]
        self.__discount_factor_gamma = discount_factor_gamma
        self.__explore_epsilon = explore_epsilon
        self.__learning_rate = learning_rate

        self.time_step = 0
        self.train_number = 0
        self.mini_batch_size = mini_batch_size
        self.mode = mode
        # initialize replay buffer
        self.replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_size)

        # initialize q_net
        self.__q_net = DuelingQNet()
        if(mode==1):
            self.__target_q_net = QNet()

    def action(self, state):
        action = 0
        if random.random() <= self.__explore_epsilon:
            action = random.randint(0, self.__action_dimension-1)
        else:
            one_state_batch = np.asarray([state])
            q_value = self.__q_net.predict(one_state_batch)
            action = torch.argmax(q_value).item()
        return action

    def train(self, max_episode, current_epsisode):
        if(self.replay_buffer.length() < self.mini_batch_size):
            return None

        # explore extent become smaller
        self.__explore_epsilon *= (1-current_epsisode/max_episode)

        # Step 1: obtain random minibatch from replay memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(
            self.mini_batch_size)

        # Step 2: calculate y
        # y_batch 是q值列表，作为标签使用
        # use original q value give by current(untrained) q_net, i.e do not train this part
        # 当前网络给出的预测值
        # y_batch = np.zeros((self.mini_batch_size, self.__action_dimension))
        y_batch = self.__q_net.predict(state_batch)

        if(self.mode==1):
            next_state_q_value_batch = self.__target_q_net.predict(
                next_state_batch)  # traget网络对next_state的q给出的预测值
        else:
            next_state_q_value_batch = self.__q_net.predict(
                next_state_batch)  # 当前网络对next_state的q给出的预测值

        for mini_batch_iter in range(self.mini_batch_size):
            if done_batch[mini_batch_iter]:
                y_batch[mini_batch_iter,
                        action_batch[mini_batch_iter]] = reward_batch[mini_batch_iter]
            else:
                y_batch[mini_batch_iter,
                        action_batch[mini_batch_iter]] = reward_batch[mini_batch_iter] + self.__discount_factor_gamma * torch.max(next_state_q_value_batch[mini_batch_iter])

        q_net_metrics = self.__q_net.train(state_batch, y_batch)

        if(self.mode==1):
            if(current_epsisode % (math.floor(0.05*max_episode)) == 0):
                self.__target_q_net.load_state_dict(self.__q_net.state_dict())

        self.train_number += 1

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        
    def reward(self,state):
        pass
