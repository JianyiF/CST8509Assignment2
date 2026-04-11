import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from aisd_examples.envs.create3_red_ball import CreateRedBallEnv

env = CreateRedBallEnv()

# Train
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

rewards = []
steps_list = []

# Test
for ep in range(100):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        steps += 1
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

    rewards.append(total_reward)
    steps_list.append(steps)

env.close()

# ===== Plot =====
def smooth(data, w=5):
    if len(data) < w:
        return data
    return np.convolve(data, np.ones(w)/w, mode='valid')

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(rewards, alpha=0.3)
plt.plot(smooth(rewards), linewidth=2)
plt.axhline(y=rewards[-1], linestyle='--')
plt.title("DQN Rewards")

plt.subplot(1,2,2)
plt.plot(steps_list, alpha=0.3)
plt.plot(smooth(steps_list), linewidth=2)
plt.title("DQN Steps")

plt.tight_layout()
plt.show()