import gymnasium as gym
import dqn
import numpy as np


# HYPERPARAMETERS
EPSILON = 1.0
MAX_EPOCHS = 10000
MAX_EPISODES = 1

# instantiate environment
env = gym.make("ALE/SpaceInvaders-v5", obs_type="grayscale", render_mode="human")
observation, info = env.reset()

# instantiate dqn agent
agent = dqn.DQN(epsilon=EPSILON, num_actions=6)

# load trained weights
agent.load()

# previous observations to tack onto the new one
prev_prev_observation = np.zeros(shape=(210, 160), dtype=np.uint8)
prev_observation = np.zeros(shape=(210, 160), dtype=np.uint8)

# main training loop
for episode in range(MAX_EPISODES):

    # start the environment
    observation, info = env.reset()

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

    # reset the environment
    env.reset()