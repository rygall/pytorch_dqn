import gymnasium as gym
import dqn
import numpy as np
import random


# instantiate dqn
agent = dqn.DQN(epsilon=0.25)

# instantiate environment
env = gym.make("ALE/SpaceInvaders-v5", obs_type="grayscale")

# define training parameters
max_epochs = 10000
max_episodes = 5
observation, info = env.reset()

# target network update frequency
target_update_freq = 50

# trackers
actions = [0, 0, 0, 0, 0, 0]
episode_reward = 0
rewards = []
final_epoch = 0
total_epochs = []

for episode in range(0, max_episodes):

    # print episode and reset environment
    print("Episode", episode)
    observation, info = env.reset()
    actions = [0, 0, 0, 0, 0, 0]

    for epoch in range(max_epochs):

        # print epoch every 10 episodes
        if epoch % 10 == 0:
            print("Epoch:", epoch, "Actions:", actions)
        
        # get next action from DQN
        action = agent.generate_action(observation)
        actions[action] += 1

        # take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # copy network to target network
        if (epoch % target_update_freq) == 0:
            agent.updateTarget()
        
        # reset the environment
        if terminated or truncated:
            final_epoch = epoch
            print("terminated/truncated")            
            break

        # train the DQN given new data
        agent.train(observation, reward)

        # update reward tracker
        episode_reward += reward
        
    # update agents gamma
    agent.setEpsilon(0.75)

    #store total rewards and total epochs
    rewards.append(episode_reward)
    total_epochs.append(final_epoch)


print("Episode Rewards:", rewards)
print("Episode Epoch:", total_epochs)


# close the environment  
env.close()