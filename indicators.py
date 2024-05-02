import pandas as pd
import numpy as np

def calculate_cci(price_data, n=20, c=0.015):
    data = price_data.copy(deep=True)
    TP = (data['high'] + data['low'] + data['close']) / 3
    CCI = pd.Series((TP - TP.rolling(n).mean()) / (c * TP.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())))), name='CCI')
    data = data.join(CCI)
    
    # Mapping CCI to signals
    data['CCI_Signal'] = 0  # Hold
    data.loc[data['CCI'] > 100, 'CCI_Signal'] = -1  # Sell
    data.loc[data['CCI'] < -100, 'CCI_Signal'] = 1  # Buy
    
    return data['CCI_Signal']

def calculate_rsi(price_data, window=14):
    data = price_data.copy(deep=True)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))

    # Mapping RSI to signals
    data['RSI_Signal'] = 0  # Hold
    data.loc[RSI > 70, 'RSI_Signal'] = -1  # Sell
    data.loc[RSI < 30, 'RSI_Signal'] = 1  # Buy
    
    return data['RSI_Signal']

def calculate_donchian(price_data, window=10):
    data = price_data.copy(deep=True)
    data['Donchian_high'] = data['high'].rolling(window).max()  
    data['Donchian_low'] = data['low'].rolling(window).min()
    
    # Mapping to signals
    data['Donchian_Signal'] = 0  # Hold
    data.loc[data['close'] >= data['Donchian_high'], 'Donchian_Signal'] = 1  # Buy
    data.loc[data['close'] <= data['Donchian_low'], 'Donchian_Signal'] = -1  # Sell
    
    return data['Donchian_Signal']

def calculate_return(price_data, short_window=21, long_window=63):
    data = price_data.copy(deep=True)
    data['1M_Return'] = data['close'].pct_change(periods=short_window) * 100
    data['3M_Return'] = data['close'].pct_change(periods=long_window) * 100
    
    # Mapping returns to signals
    # This is a bit more subjective and can be adjusted based on your criteria
    data['Return_Signal'] = 0  # Hold
    data.loc[data['1M_Return'] > 0, 'Return_Signal'] = 1  # Buy
    data.loc[data['1M_Return'] < 0, 'Return_Signal'] = -1  # Sell
    
    return data['Return_Signal']


