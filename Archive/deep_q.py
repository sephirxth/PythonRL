import tensorflow as tf
from tensorflow import keras
import numpy as np
# import matplotlib.pyplot as matplot
# import nni
import random
import replay_buffer


class DeepQ(object):

    # __q_net = tf.keras.Sequential()
    # __q_net_target = tf.keras.Sequential()
    # replay_buffer_size = 0
    # replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_size)
    # __action_dimension = 0
    # __state_dimension = 0
    # __explore_epsilon = 0
    # __learning_rate_alpha = 0
    # __discount_factor_gamma = 0
    # time_setp = 0
    # mini_batch_size = 0
    # train_number = 0

    def __init__(self, env=None, replay_buffer_size=10000, learning_rate=0.01, explore_epsilon=0.5, discount_factor_gamma=0.9, batch_size=32):
        self.__action_dimension = env.action_space.n
        self.__state_dimension = env.observation_space.shape[0]
        self.__discount_factor_gamma = discount_factor_gamma
        self.__explore_epsilon = explore_epsilon
        self.__learning_rate = learning_rate

        self.time_step = 0
        self.train_number = 0
        self.mini_batch_size = batch_size

        # initialize replay buffer
        self.replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_size)

        # initialize q_net
        self.__q_net = keras.Sequential([
            keras.layers.Dense(32, activation='relu',
                               input_shape=(self.__state_dimension,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.__action_dimension, name="q_net")
        ])
        self.__optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

        self.__q_net.compile(
            optimizer=keras.optimizers.RMSprop(lr=self.__learning_rate),
            # optimizer='adam',
            # optimizer="SGD",
            # optimizer="RMSprop",
            loss='MSE',
            metrics=['mae'])

        # q_target_net
        # inputs = tf.keras.Input(
        #     shape=(self.__state_dimension,), name='state_layer')
        # x = tf.keras.layers.Dense(32, activation='relu')(inputs)
        # x = tf.keras. layers.Dense(32, activation='relu')(x)
        # outputs = tf.keras.layers.Dense(self.__action_dimension)(x)

        # model = keras.Model(inputs=inputs, outputs=outputs,
        #                     name='q_net_target')
        # self.__q_net_target = model
        # # 编译为静态图，提升计算性能
        # self.__q_net_target.compile(optimizer='adam',
        #                             loss='MSE',
        #                             metrics=['accuracy'])

    # action sequence, 0, 1, 2, ...

    def action(self, state):
        action = 0
        if random.random() <= self.__explore_epsilon:
            action = random.randint(0, self.__action_dimension-1)
        else:
            one_state_batch = np.asarray([state])
            q_value = self.__q_net.predict(one_state_batch)
            tf.keras.backend.clear_session()
            action = np.argmax(q_value)
            
        return action

    def train(self, max_episode, current_epsisode):
        if(self.replay_buffer.length() < self.mini_batch_size):
            return None

        self.__explore_epsilon *= (1-current_epsisode/max_episode)

        # Step 1: obtain random minibatch from replay memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(
            self.mini_batch_size)

        # Step 2: calculate y
        # y_batch 是q值列表，作为标签使用
        # use original q value give by current(untrained) q_net, i.e do not train this part
        # 当前网络给出的预测值
        # y_batch = np.zeros((self.mini_batch_size, self.__action_dimension))
        y_batch = self.__q_net.predict(state_batch).copy()

        next_state_q_value_batch = self.__q_net.predict(
            next_state_batch)  # 当前网络对next_state的q给出的预测值

        tf.keras.backend.clear_session()
        
        for mini_batch_iter in range(self.mini_batch_size):
            if done_batch[mini_batch_iter]:
                y_batch[mini_batch_iter,
                        action_batch[mini_batch_iter]] = reward_batch[mini_batch_iter]
            else:
                y_batch[mini_batch_iter,
                        action_batch[mini_batch_iter]] = reward_batch[mini_batch_iter] + self.__discount_factor_gamma * np.max(next_state_q_value_batch[mini_batch_iter])
      
       

        q_net_metrics = self.__q_net.train_on_batch(state_batch, y_batch)

        self.train_number += 1

        # if(self.train_number % 100 == 0):
        #     self.__q_net_target = self.__q_net
        #     print("No.%d train q_net metrics = " %
        #           self.train_number, q_net_metrics)

    def load(self):
        return

    def save(self):
        return 0

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
