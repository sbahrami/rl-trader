import os
import random
from abc import ABC, abstractmethod
import gym
from gym import spaces
import numpy as np
import pandas as pd

import stock_analysis
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, stock_price, signal_data, initial_balance, initial_share, random_start=False):
        super(StockTradingEnv, self).__init__()
        self.log = []
        self.rewards = []
        self.signal_data = signal_data
        self.n_step = self.signal_data.shape[0]
        self.stock_price = stock_price
        self.random_start = random_start
        
        # Define action space: 0 = sell, 1 = hold, 2 = buy
        self.action_space = spaces.Discrete(3)

        # Define the observation space to include signal data
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.signal_data.shape[1]+2,), dtype=np.float32)
        self.initial_balance = initial_balance
        self.initial_share = initial_share
        self.current_step = 0
        self.current_balance = initial_balance
        self.shares_held = initial_share
        self.current_portfolio_value = initial_balance

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        if self.current_step >= self.n_step-1:
            done = True
        else:
            done = False
        
        reward = self._calculate_reward()
        obs = self._next_observation()
        obs = np.reshape(obs, [1, obs.shape[0]])


        # Log the necessary details
        self.log.append({
            'time_step': self.current_step,
            'close_price': self.stock_price.iloc[self.current_step]['close'],
            'num_shares': self.shares_held,
            'balance': self.current_balance,
            'action': action
        })
        return obs, reward, done, {}
    
    def get_logs(self):
        return pd.DataFrame(self.log)
    
    def get_rewards_log(self):
        return self.rewards

    def reset(self):
        # Reset the state of the environment to an initial state
        self.log = []
        self.rewards = []
        
        if self.random_start:
            self.current_step = random.randint(0, len(self.stock_price)-2)
        else:
            self.current_step = 0
        
        self.current_balance = self.initial_balance
        self.shares_held = self.initial_share
        self.current_portfolio_value = self.initial_balance
        return self._next_observation()

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = self.stock_price.iloc[self.current_step]['open']
        action_type = action
        amount = 1  # Number of shares to buy/sell
        
        if action_type == 0:
            # Sell amount shares
            if self.shares_held > 0:
                self.current_balance += current_price * amount
                self.shares_held -= amount
        elif action_type == 2:
            # Buy amount shares
            if self.current_balance > current_price:
                self.current_balance -= current_price * amount
                self.shares_held += amount

    def _next_observation(self):
        # Get the stock data for the current step
        frame = self.signal_data.iloc[self.current_step]
        current_price = self.stock_price.loc[self.current_step, 'close']

        # Append the normalized current state to the observation
        obs = np.append(frame.values, [self.current_balance/current_price, self.shares_held/self.initial_share])

        return obs

    def _calculate_reward(self):
        # Simple reward function: profit or loss of the portfolio
        future_period = 20
        current_price = self.stock_price.at[self.current_step, 'close']

        if self.current_step + future_period < len(self.stock_price):
            # Calculate the mean of the next 30-day prices
            future_prices = self.stock_price.loc[self.current_step+1:self.current_step+future_period+1, 'close']
            mean_future_price = future_prices.mean()
        else:
            mean_future_price = current_price

        # if self.current_step < len(self.stock_price) - 1:
        #     mean_future_price = self.stock_price.loc[self.current_step+1, 'close']
        # else:
        #     mean_future_price = current_price

        reward = 20 * self.shares_held * (mean_future_price - current_price) / current_price
        self.rewards.append(reward)

        current_portfolio_value = self.current_balance + self.shares_held * current_price
        self.current_portfolio_value = current_portfolio_value
        
        return reward

    def render(self, mode='human', close=False):
        if close:
            plt.close()
            return

        # Ensure that we only initialize the plot once
        if not hasattr(self, 'figure') or self.figure is None:
            self.figure, self.ax = plt.subplots()
            plt.show(block=False)

        self.ax.clear()
        self.ax.plot(self.stock_price['close'][:self.current_step + 1])
        self.ax.set_title('Stock Price')

        # Highlight buy actions in green and sell actions in red
        for log in self.log:
            color = 'g' if log['action'] == 2 else 'r' if log['action'] == 0 else 'k'
            self.ax.plot(log['time_step'], self.stock_price['close'][log['time_step']], color + 'o')
        plt.pause(0.01)

class Trader(ABC):
    @abstractmethod
    def act(self, state):
        pass

class DQNAgent(Trader):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon_i = 0.5  # exploration rate
        self.epsilon_f = 0.1
        self.epsilon = self.epsilon_i
        # self.epsilon_decay = 0.9999
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def clear_memory(self):
        self.memory = deque(maxlen=2000)

    def _build_model(self):
        # Define the neural network model
        model = nn.Sequential(
            nn.Linear(self.state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state.squeeze(), action, reward, next_state.squeeze(), done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.from_numpy(state).float().unsqueeze(0)  # Convert state to tensor
        act_values = self.model(state).squeeze() 
        action = torch.argmax(act_values, dim=0).item()  # Returns the action with the highest Q-value
        return action
    

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)
            
            if done:
                target = reward
            else:
                target = reward + self.gamma * torch.max(self.model(next_state).detach())
            
            current_q = self.model(state).squeeze()[action]
            loss = (current_q - target).pow(2).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # if self.epsilon > self.epsilon_f:
        #     self.epsilon *= self.epsilon_decay

def save_agent_state(agent, filename="agent_checkpoint.pth"):
    checkpoint = {
        'model_state_dict': agent.model.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_agent_state(agent, filename="agent_checkpoint.pth"):
    checkpoint = torch.load(filename)
    agent.model.load_state_dict(checkpoint['model_state_dict'])

class RSIAgent(Trader):
    def act(self, state):
        action = state.squeeze()[0] + 1
        return action
    
class CCIAgent(Trader):
    def act(self, state):
        action = state.squeeze()[1] + 1
        return action
    
class DonchianAgent(Trader):
    def act(self, state):
        action = state.squeeze()[2] + 1
        return action
    
class ReturnAgent(Trader):
    def act(self, state):
        action = state.squeeze()[3] + 1
        return action

def read_input_files(file_names):
    dfs = []
    for file_name in file_names:
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            df['date'] = pd.to_datetime(df['date'])
            dfs.append(df)
        else:
            print(f"The input file {file_name} does not exit in the current folder.")
        merged_df = dfs[0]  # Start with the first DataFrame

        for df in dfs[1:]:
            # Perform an as-of merge with each subsequent DataFrame
            merged_df = pd.merge_asof(merged_df.sort_values('date'), df.sort_values('date'), on='date', direction='nearest') #, tolerance=pd.Timedelta('10 days')

        merged_df.set_index('date', drop=True, inplace=True)

    return merged_df

def plot_time_log(logs, file_name):

    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(10, 8))  # 5 rows, 1 column, shared x-axis

    # Plot closing price
    axs[0].plot(logs['time_step'], logs['close_price'], label='Close Price', color='tab:blue')
    axs[0].set_ylabel('Close Price')
    axs[0].legend(loc='upper left')
    axs[0].grid(True)

    # Plot number of shares
    axs[1].plot(logs['time_step'], logs['num_shares'], label='Number of Shares', color='tab:orange')
    axs[1].set_ylabel('Shares')
    axs[1].legend(loc='upper left')
    axs[1].grid(True)

    # Plot balance
    axs[2].plot(logs['time_step'], logs['balance'], label='Balance', color='tab:green')
    axs[2].set_ylabel('Balance')
    axs[2].legend(loc='upper left')
    axs[2].grid(True)

    # Plot portfolio
    portfolio = logs['balance'] + logs['num_shares'] * logs.at[len(logs)-1, 'close_price']
    axs[3].plot(logs['time_step'], portfolio, label='Portfolio', color='tab:green')
    axs[3].set_ylabel('Total Portfolio Value')
    axs[3].legend(loc='upper left')
    axs[3].grid(True)

    # Plot actions
    axs[4].step(logs['time_step'], logs['action'], label='Actions', color='tab:red', where='mid')
    axs[4].set_yticks([0, 1, 2])
    axs[4].set_yticklabels(['Sell', 'Hold', 'Buy'])
    axs[4].set_ylabel('Actions')
    axs[4].set_xlabel('Time Step')
    axs[4].legend(loc='upper left')
    axs[4].grid(True)

    plt.suptitle('Trading Log')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the padding to leave space for the suptitle
    fig.savefig(file_name)
    plt.show()

def plot_benchmark_log(logs_list, file_name):
    # Create a matplotlib color cycle for plotting multiple lines
    color_cycle = plt.cm.tab10.colors  # Default color cycle from matplotlib

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 8))  # 5 rows, 1 column, shared x-axis
    labels = ['DQN RL', 'RSI', 'CCI', 'Donchian', 'Return']
    linewidths = [3, 1, 1, 1, 1]

    # Plot closing price
    logs = logs_list[0]
    axs[0].plot(logs['time_step'], logs['close_price'], label='Close Price')
    axs[0].set_ylabel('Close Price')
    axs[0].legend(loc='upper left')
    axs[0].grid(True)

    # Iterate through each dictionary in the list
    for index, logs in enumerate(logs_list):
        color = color_cycle[index % len(color_cycle)]  # Cycle through colors

        # Plot number of shares
        axs[1].plot(logs['time_step'], logs['num_shares'], label=labels[index], color=color, linewidth=linewidths[index])
        axs[1].set_ylabel('Number of Shares')
        axs[1].legend(loc='upper left')
        axs[1].grid(True)

        # Plot portfolio value
        portfolio = logs['balance'] + logs['num_shares'] * logs['close_price']
        axs[2].plot(logs['time_step'], portfolio, label=labels[index], color=color, linewidth=linewidths[index])
        axs[2].set_ylabel('Total Portfolio Value')
        axs[2].legend(loc='upper left')
        axs[2].grid(True)

        # Plot actions
        axs[3].step(logs['time_step'], logs['action'], label=labels[index], color=color, where='mid', linewidth=linewidths[index])
        axs[3].set_yticks([0, 1, 2])
        axs[3].set_yticklabels(['Sell', 'Hold', 'Buy'])
        axs[3].set_ylabel('Actions')
        axs[3].set_xlabel('Time Step')
        axs[3].legend(loc='upper left')
        axs[3].grid(True)

    plt.suptitle('Trading Log')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the padding to leave space for the suptitle
    fig.savefig(file_name)
    plt.show()

def train_agent(env, agent, n_ep, batch_size, reset_pr):
    rewards_stds = []
    rewards_means = []
    agent.epsilon_decay = (agent.epsilon_f/agent.epsilon_i)**(1/n_ep)
    plt.ion()
    fig, ax = plt.subplots()
    for e in range(n_ep):
        state_size = env.observation_space.shape[0]
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            random_reset = (np.random.rand() < reset_pr)
            if random_reset:
                break
            state = next_state
            if done:
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        rewards_log = env.get_rewards_log()
        rewards_mean = np.mean(rewards_log)
        rewards_std= np.std(rewards_log)
        rewards_means.append(rewards_mean)
        rewards_stds.append(rewards_std)
        print(f"Episode: {e}\t Rewards Mean: {rewards_mean:.2f}\t Rewards STD: {rewards_std:.2f}\t Epsilon: {agent.epsilon:.2f}")
        if agent.epsilon > agent.epsilon_f:
            agent.epsilon *= agent.epsilon_decay
        
        # Update plot for every episode
        if e % 1 == 0:  # You can change the number to update after more episodes
            # clear_output(wait=True)
            
            # smooth_window = max(int(e/100), 1)  # Avoid zero window size
            ax.errorbar(range(len(rewards_means)), rewards_means, yerr=rewards_stds, ecolor='gray', color='blue')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Rewards Mean')
            ax.set_title('Rewards Mean and STD of each Episode')
            # plt.draw()
            plt.pause(0.1)
            # plt.clf()
            # plt.show()
    plt.ioff()

    # Save the final plot
    fig.savefig('figures/train_portfolio_vs_ep.pdf')
    return agent

def test_agent(agent, env):
    state_size = env.observation_space.shape[0]
    agent.epsilon = 0
    agent.learning_rate = 0
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    logs = []
    while not done:  # or env.spec.max_episode_steps
        # print("\033[A\033[K", end="")
        # print(f"Time step: {time_step}") 
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        # env.render()
    logs.append(env.get_logs())
    
    benchmark_agents = [RSIAgent(), CCIAgent(), DonchianAgent(), ReturnAgent()]
    for signal_agent in benchmark_agents:
        state = env.reset()
        done = False
        while not done:  # or env.spec.max_episode_steps
            # print("\033[A\033[K", end="")
            # print(f"Time step: {time_step}") 
            action = signal_agent.act(state)
            next_state, _, done, _ = env.step(action)
            state = next_state
        logs.append(env.get_logs())

    # logs = env.get_logs() 
    plot_time_log(logs[0], file_name='figures/test_log.pdf')
    plot_benchmark_log(logs, 'figures/benchmark_log.pdf')