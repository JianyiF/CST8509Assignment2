import numpy as np
import matplotlib.pyplot as plt
from aisd_examples.envs.create3_red_ball import CreateRedBallEnv

# Create env
env = CreateRedBallEnv()

num_states = env.observation_space.n
num_actions = env.action_space.n

# Q-table
qtable = np.zeros((num_states, num_actions))

# Hyperparameters
episodes = 100
alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.98
epsilon_min = 0.05

rewards = []
steps_list = []

# Training
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        steps += 1

        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(qtable[state])

        next_state, reward, done, _, _ = env.step(action)

        qtable[state][action] += alpha * (
            reward + gamma * np.max(qtable[next_state]) - qtable[state][action]
        )

        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    rewards.append(total_reward)
    steps_list.append(steps)

    print(f"Episode {episode+1}/{episodes} | Steps: {steps} | Reward: {total_reward}")

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
plt.title("Q-Learning Rewards")

plt.subplot(1,2,2)
plt.plot(steps_list, alpha=0.3)
plt.plot(smooth(steps_list), linewidth=2)
plt.title("Q-Learning Steps")

plt.tight_layout()
plt.show()