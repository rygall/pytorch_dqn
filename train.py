import dqn
import gymnasium as gym
import numpy as np

import os
from datetime import datetime
import matplotlib.pyplot as plt


# HYPERPARAMETERS
EPSILON = 0.9
LR = 0.0001
GAMMA = 0.3
TARGET_NET_UPDATE_FREQ = 30
MAX_EPOCHS = 10000
MAX_EPISODES = 1000


# instantiate environment
env = gym.make("ALE/SpaceInvaders-v5", obs_type="grayscale")
observation, info = env.reset()

# instantiate dqn agent
agent = dqn.DQN(epsilon=EPSILON, lr=LR, gamma=GAMMA, update_freq=TARGET_NET_UPDATE_FREQ, num_actions=6)

# load trained weights
# agent.load()

# record training session time and date
now = datetime.now()
year = now.strftime("%Y")
month = now.strftime("%m")
day = now.strftime("%d")
time = now.strftime("%H%M%S")
date_time = now.strftime("  [%m_%d_%Y @ %H_%M_%S]")

# create folder to save training session information
dir_name = "results//Training Session" + str(date_time)
fig_dir_name = dir_name + "//figures"
weights_dir_name = dir_name + "//weights"
os.mkdir(dir_name)
os.mkdir(fig_dir_name)
os.mkdir(weights_dir_name)

# total reward tracker
rewards = []

# previous observation to tack onto the new one
prev_prev_observation = np.zeros(shape=(210, 160), dtype=np.uint8)
prev_observation = np.zeros(shape=(210, 160), dtype=np.uint8)

# main training loop
for episode in range(MAX_EPISODES):

    # start the environment
    observation, info = env.reset()

    # episode total reward
    episode_total_reward = 0

    for epoch in range(MAX_EPOCHS):   

        # get action selection from DQN
        markov_state = np.stack((prev_prev_observation, prev_observation, observation), axis=0)
        action = agent.select_action(markov_state)

        # save the previous observation
        prev_prev_observation = prev_observation
        prev_observation = observation

        # take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # reset the environment
        if terminated or truncated:
            final_epoch = epoch        
            break
        
        # increment episode total reward
        episode_total_reward += reward

        # update the dqn
        agent.train(markov_state, reward, epoch)
    
    # save total episode reward
    rewards.append(episode_total_reward)

    # reset the environment
    env.reset()

# save plot
plot_name = fig_dir_name + "//Total Reward vs Episode.png"
plt.plot(rewards)
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.savefig(plot_name)
plt.close(plot_name)

# save agent weights
agent.save(weights_dir_name + "//dqn.pth")

