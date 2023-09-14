import gymnasium as gym
import dqn
import numpy as np

# instantiate dqn
agent = dqn.DQN(epsilon=1)
agent.load()

# instantiate environment
env = gym.make("ALE/SpaceInvaders-v5", render_mode="human", obs_type="grayscale")
observation, info  = env.reset()
env.seed = 0
terminated = False
truncated = False

# run one episode
while not truncated or not terminated:   
    
    # get next action from DQN
    action = agent.generate_action(observation)
    
    # take a step in the environment
    observation, reward, terminated, truncated, info = env.step(action)
        
    # break if the environment terminates or truncates
    if terminated or truncated:
        observation, info = env.reset()
        break

env.close()