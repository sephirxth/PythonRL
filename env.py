import gym

# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 10000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode

gloal_env = gym.make(ENV_NAME)
global_next_state=[]
global_reward=0
global_done=False

def plusOne(a):
    print("feed arg: ", a)
    return a+1

def init_state():
    init_state=gloal_env.reset()
    return init_state

def get_global_states():
    return global_next_state

def env_step(action):
    global_next_state, global_reward, global_done, _info = gloal_env.step(action)
    return [ global_next_state, global_reward, global_done]

def get_global_reward():
    return global_reward

def get_global_done():
    return global_done

def env_close():
    gloal_env.close()

