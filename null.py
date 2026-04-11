import gymnasium as gym

from fan00079_aisd_examples import aisd_examples

env = gym.make("aisd_examples/CreateRedBall-v0")

obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

env.close()