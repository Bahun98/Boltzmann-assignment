import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import stats

# hyper parameters
alpha = 0.1  # Learning rate
gamma = 0.95  # Discount factor between (0-1)
episodes = 3 * 2000  # Number of episodes to learn, keep this a multiple of four for nice plotting
T = 100  # Maximum steps in an episode
epsilon = 1  # Exploration rate between (0-1)
cost_of_living = -0.01  # Used when FrequentRewards = True, incentivize the agent for efficiency

# Choose environment
env = gym.make("FrozenLake8x8-v1", is_slippery=False)
FrequentRewards = False  # When False the original environment rewards are used

Q = np.zeros([env.observation_space.n, env.action_space.n])
rewards_per_episode = []
q_values_at_intervals = []  # Store Q-values at intervals
epsTillSucc = []

for i in range(10):
    breaker = False
    for episode in range(episodes):
        state,prob = env.reset()
        total_reward = 0
        episode_states = []
        episode_actions = []
        episode_rewards = []
        #epsilon *= .998
        
        for step in range(T):
            # Choose action based on epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax(Q[state, :])

            new_state, reward, terminated,truncated, info = env.step(action)
            if reward == 1:
                epsTillSucc.append(episode)

            if FrequentRewards:
                if terminated and reward == 0: # agent fall in the hole!
                    reward = reward - 1
                reward = reward + cost_of_living

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            total_reward += reward
            state = new_state

            if terminated:
                if reward > 0:
                    epsTillSucc.append(episode)
                    print(f"Episode {episode} finished after {step + 1} steps. Epsilon is {epsilon}, LR {alpha} Success!")
                    breaker = True
                break

        rewards_per_episode.append(total_reward)

        # Calculate returns and update Q-values
        G = 0
        for t in range(len(episode_states) - 1, -1, -1):
            state = episode_states[t]
            action = episode_actions[t]
            reward = episode_rewards[t]

            G = gamma * G + reward
            Q[state][action] += alpha * (G - Q[state][action])

        # Store Q-values at intervals (e.g., every 100 episodes)
        if (episode + 1) % (episodes // 4) == 0:
            q_values_at_intervals.append(np.copy(Q))  # Store a copy of Q-values
        
        #break in case we reached the goal
        if breaker == True:
            break

print(epsTillSucc)

# Calculate mean and standard deviation of the epsTillSucc
mean = np.mean(epsTillSucc)
print(f"\nMean is {mean}")
std_dev = np.std(epsTillSucc)  # Use ddof=1 for sample standard deviation

# Set the confidence level (e.g., 95% confidence interval)
confidence_level = 0.95

# Calculate the standard error (standard deviation divided by the square root of sample size)
std_error = std_dev / np.sqrt(len(epsTillSucc))

# Calculate the margin of error using the t-distribution (for small sample sizes)
# Degrees of freedom = n - 1
degrees_of_freedom = len(epsTillSucc) - 1
t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
margin_of_error = t_value * std_error

# Calculate the confidence interval
lower_bound = mean - margin_of_error
upper_bound = mean + margin_of_error

print(f"Margin of error: {margin_of_error}")
print(f"Confidence Interval ({confidence_level * 100}%): ({lower_bound}, {upper_bound})")