import matplotlib.pyplot as plt
import pandas as pd
from trader_utils import *

def main():
    # Read input data
    file_names = ['spy_price_signals.csv']
    data = read_input_files(file_names)
    data.sort_values(by='date', ascending=True, inplace=True)

    ##################### Training phase #####################

    # Trainer environment setup
    dates = ['2022-01', '2023-01']
    signal_types = ['rsi_sig', 'cci_sig', 'donchian_sig', 'return_sig']
    train_signal_data = (
        data[(data.index >= dates[0]) & (data.index < dates[1])]
        .loc[:, signal_types]
        .reset_index(drop=True)
    )

    train_price_data = (
        data[(data.index >= dates[0]) & (data.index < dates[1])]
        .loc[:, ['close', 'open']]
        .reset_index(drop=True)
    )
    env = StockTradingEnv(train_price_data, train_signal_data, initial_balance=10000, initial_share=10, random_start=True)

    # Agent setup
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]
    agent = DQNAgent(state_size, action_size)
    # load_agent_state(filename='agent_checkpoint.pth')

    # Train the agent
    agent = train_agent(env, agent, n_ep=2000, batch_size=32, reset_pr=0.01)
    save_agent_state(agent)

    ##################### Testing phase #####################

    # Test environment setup
    data.sort_values(by='date', ascending=True, inplace=True)
    dates = ['2022-01', '2023-01']
    test_signal_data = (
        data[(data.index >= dates[0]) & (data.index < dates[1])]
        .loc[:, signal_types]
        .reset_index(drop=True)
    )
    test_price_data = (
        data[(data.index >= dates[0]) & (data.index < dates[1])]
        .loc[:, ['close', 'open']]
        .reset_index(drop=True)
    )
    env = StockTradingEnv(test_price_data, test_signal_data, initial_balance=10000, initial_share=10, random_start=False)

    test_agent(agent, env)

if __name__ == '__main__':
    main()