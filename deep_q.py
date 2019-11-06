import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as matplot
import nni
import random


class DeepQ(object):

    __q_net = tf.keras.Sequential()
    __q_net_target = tf.keras.Sequential()
    __replay_buffer_size = 0
    __replay_buffer = ReplayBuffer(__replay_buffer_size)
    __action_dimension = 0
    __state_dimension = 0
    __explore_epsilon = 0
    __learning_rate_alpha = 0
    __discount_factor_gamma = 0

    

    def __init__(self, replay_buffer_size):
        # initialize replay buffer
        self.__replay_buffer = ReplayBuffer(replay_buffer_size)

        # initialize q_net
        self.__q_net = tf.keras.Sequential()
        self.__q_net.add(keras.layers.Dense(32, activation='relu'))
        self.__q_net.add(keras.layers.Dense(32, activation='relu'))
        self.__q_net.add(keras.layers.Dense(3))

        # 编译为静态图，提升计算性能
        self.__q_net.compile(optimizer='adam',
                             loss='MSE',
                             metrics=['accuracy'])

        # initialize q_net_target
        self.__q_net_target = tf.keras.Sequential()
        self.__q_net_target.add(keras.layers.Dense(32, activation='relu'))
        self.__q_net_target.add(keras.layers.Dense(32, activation='relu'))
        self.__q_net_target.add(keras.layers.Dense(3))

        self.__q_net_target.compile(optimizer='adam',
                                    loss='MSE',
                                    metrics=['accuracy'])
    # action sequence, 0, 1, 2, ...

    def give_action(self, state):
        if random.random() <= self.__explore_epsilon:
            action = random.randint(0, self.__action_dimension-1)
        else:
            q_value = self.__q_net.predict(state)
            action = np.argmax(q_value)
        return action

    def train(self, batch_size):
        # Step 1: obtain random minibatch from replay memory
        minibatch = self.__replay_buffer.sample(batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []  # y_batch 是q值列表，作为标签使用
        q_value_batch = self.__q_net.predict(next_state_batch)  # 当前网络给出的预测值
        for i in range(0, batch_size):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] +
                               self.__discount_factor_gamma * np.max(q_value_batch[i]))

        self.__q_net.train_on_batch(state_batch, y_batch)
        # 以下可能不需要
        #
        # self.optimizer.run(feed_dict={
        #     self.y_input: y_batch,
        #     self.action_input: action_batch,
        #     self.state_input: state_batch
        # })
        # return

    def load(self):
        return

    def save(self):
        return 0


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.

        Retrun
        ----------
        (array([ state_t, ...]), 
        array([action, ...]), 
        array([reward, ...]), 
        array([state_t+1, ...]), 
        array([done, ...]))

        columns: batch_size
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data  # 从开始时覆盖, self_next_idx=0
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes)
