import random


class BasicQ(object):
    def action(self, state):
        if random.random() <= self.__explore_epsilon:
            action = random.randint(0, self.__action_dimension-1)
        else:
            one_state_batch = np.asarray([state])
            q_value = self.__q_net.predict(one_state_batch)
            action = np.argmax(q_value)
        return action
