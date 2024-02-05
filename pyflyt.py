import gymnasium
import PyFlyt.gym_envs

env = gymnasium.make("PyFlyt/Fixedwing-Waypoints-v1", render_mode="human")

term, trunc = False, False
obs, _ = env.reset()

for _ in range(1000):

    obs, rew, term, trunc, _ = env.step(env.action_space.sample())

    if term or trunc:
        observation, info = env.reset()

env.close()