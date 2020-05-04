import gym
import torch

from PytorchDeepQ import DeepQ as pytorchDeepQ

# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 10000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode



def testModel():
     # initialize OpenAI Gym env and dqn agent
    print("now test function run")
    print("feed arg: ", a)
    env = gym.make(ENV_NAME)
    # agent = pytorchDeepQ.DeepQ(env)

    agent = torch.load("D:\working\WorkingInEnhancedAIMSUNPlatform\LaneChanging\microSDK_EnhancedAimsun\CooperativeLaneChangingModel\data\model_200_save.pt")
   
    
    print("now run save/load TEST")
    total_reward = 0
    for i in range(TEST):
        state = env.reset()
        
        for j in range(STEP):
            # env.render()
            print("now test function run step x")
            action = agent.action(state)  # direct action for test
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        ave_reward = total_reward/TEST
        print('TEST save/load model, Evaluation Average Reward:', ave_reward)
    env.close()
    return 0

def trainModel():
    
    print("now train function run")
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    # agent = deep_q.DeepQ(env)
    agent = pytorchDeepQ(env=env)

    for episode in range(EPISODE):
        print("Episode = %d" % episode)
        # initialize task

        state = env.reset()

        # Train
        for step in range(STEP):
            action = agent.action(state)   # e-greedy action for train
            next_state, reward, done, info = env.step(action)
          
            # Define reward for agent
            reward = -1 if done else 0.1
            agent.remember(state, action, reward, next_state, done)
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
                    # env.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break

            ave_reward = total_reward/TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
            env.close()
            if ave_reward >= 200:
                torch.save(agent, "D:\working\WorkingInEnhancedAIMSUNPlatform\LaneChanging\microSDK_EnhancedAimsun\CooperativeLaneChangingModel\data\model_200_save.pt")
                break

    env.close()
    return 0

trainModel()