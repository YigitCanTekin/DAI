import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, wilcoxon

# Load the stock price data
tsla_file_path = 'C:\\Users\\ototr\\Desktop\\TSLA_stock_data.csv'
cydy_file_path = 'C:\\Users\\ototr\\Desktop\\CYDY_stock_data.csv'
pnfp_file_path = 'C:\\Users\\ototr\\Desktop\\PNFP_stock_data.csv'

tsla_data = pd.read_csv(tsla_file_path)
cydy_data = pd.read_csv(cydy_file_path)
pnfp_data = pd.read_csv(pnfp_file_path)

# Ensure date columns are in datetime format
def preprocess_data(stock_df):
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df['return'] = stock_df['Close'].pct_change()
    return stock_df

tsla_data = preprocess_data(tsla_data)
cydy_data = preprocess_data(cydy_data)
pnfp_data = preprocess_data(pnfp_data)

# Event dates
events = {
    'TSLA_illegal': pd.to_datetime('2018-08-07'),
    'TSLA_sec': pd.to_datetime('2018-09-30'),
    'PNFP_illegal': pd.to_datetime('2016-01-05'),
    'PNFP_sec': pd.to_datetime('2016-10-21'),
    'CYDY_illegal': pd.to_datetime('2020-04-01'),
    'CYDY_sec': pd.to_datetime('2022-12-20')
}

# Function to calculate CAR
def calculate_car(event_date, stock_df, event_window=30):
    # Ensure event_date is in the stock_df
    stock_df = stock_df.set_index('Date')
    if event_date not in stock_df.index:
        print(f"Event date {event_date} not found in stock data.")
        return None
    event_index = stock_df.index.get_loc(event_date)
    if event_index - event_window < 0 or event_index + event_window >= len(stock_df):
        print(f"Event window for {event_date} is out of bounds.")
        return None
    window_df = stock_df.iloc[event_index-event_window:event_index+event_window+1].copy()
    window_df['CAR'] = window_df['return'].cumsum()
    window_df = window_df.reset_index()
    return window_df

# Calculate CAR for each event
car_data = {
    'TSLA_illegal': calculate_car(events['TSLA_illegal'], tsla_data),
    'TSLA_sec': calculate_car(events['TSLA_sec'], tsla_data),
    'PNFP_illegal': calculate_car(events['PNFP_illegal'], pnfp_data),
    'PNFP_sec': calculate_car(events['PNFP_sec'], pnfp_data),
    'CYDY_illegal': calculate_car(events['CYDY_illegal'], cydy_data),
    'CYDY_sec': calculate_car(events['CYDY_sec'], cydy_data)
}

# Plot CAR for each event
for event in car_data:
    if car_data[event] is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(car_data[event]['Date'], car_data[event]['CAR'], label=f'CAR - {event}')
        plt.title(f'Cumulative Abnormal Returns for {event}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Abnormal Returns')
        plt.legend()
        plt.grid(True)
        plt.show()

# Function to perform statistical tests
def perform_statistical_tests(car_df):
    if car_df is None:
        return np.nan, np.nan, np.nan, np.nan
    returns = car_df['return'].dropna()
    t_stat, t_p_value = ttest_1samp(returns, 0)
    sign_stat, sign_p_value = wilcoxon(returns)
    return t_stat, t_p_value, sign_stat, sign_p_value

# Perform statistical tests for each event
for event in car_data:
    t_stat, t_p_value, sign_stat, sign_p_value = perform_statistical_tests(car_data[event])
    print(f"{event}:")
    print(f"  t-test: t-statistic = {t_stat}, p-value = {t_p_value}")
    print(f"  Wilcoxon signed-rank test: statistic = {sign_stat}, p-value = {sign_p_value}")
