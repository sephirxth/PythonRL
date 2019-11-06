import replay_buffer
import numpy as np
test = replay_buffer.ReplayBuffer(7)
test.add(1, 1, 1, 1, 1)
test.add(2, 2, 2, 2, 2)
test.add(3, 3, 3, 3, 3)
test.add(4, 4, 4, 4, 4)
test.add(5, 5, 5, 5, 5)
test.add(6, 6, 6, 6, 6)
test.add(7, 7, 7, 7, 7)


minibatch = test.sample(3)
print("minibatch is :",minibatch)
state_batch = minibatch[0:][0]
print("state_batch is :",state_batch)
action_batch = [data[1] for data in minibatch]
print(action_batch)
