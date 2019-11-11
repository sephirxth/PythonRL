import gym

import deep_q

# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 10000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    agent = deep_q.DeepQ(env)
    state, action, reward, next_state, done, info = 0, 0, 0, 0, 0, 0
    for episode in range(EPISODE):
        print("Episode = %d" % episode)
        # initialize task

        state = env.reset()
        # Train

        for step in range(STEP):
            action = agent.action(state)  # e-greedy action for train
            # action = 1
            next_state, reward, done, info = env.step(action)

            # Define reward for agent
            reward = -1 if done else 0.1
            agent.remember(state, action, reward, next_state, done)
            if(agent.replay_buffer.length() > agent.mini_batch_size):
                agent.train(EPISODE, episode)

            state = next_state
            if done:
                break
        # Test every 100 episodes
        if episode % 100 == 0:
            print("now run TEST")
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break

            ave_reward = total_reward/TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
            env.close()
            if ave_reward >= 200:
                break

    env.close()


main()
