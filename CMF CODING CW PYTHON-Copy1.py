#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib as mpl


# In[2]:


initial_data= yf.download("AAPL",start="2022-04-04", end="2024-04-04")


# In[3]:


initial_data.head()


# In[5]:


# Importing necessary libraries
import matplotlib.pyplot as plt

# Loading the data from the CSV file
apple_data = pd.read_csv('apple_data.csv', index_col=0, parse_dates=True)

# Plotting the equity price movement
plt.figure(figsize=(10, 6))
plt.plot(apple_data['Adj Close'], label='Apple Stock Price', color='blue')
plt.title('Apple Inc. (AAPL) Stock Price Movement (April 4, 2022 - April 4, 2024)')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Calculating daily returns
daily_returns = apple_data['Adj Close'].pct_change()

# Calculating annualized average return
average_daily_return = daily_returns.mean()
annualized_average_return = average_daily_return * 252  # 252 trading days in a year

# Calculating annualized standard deviation
standard_deviation = daily_returns.std()
annualized_standard_deviation = standard_deviation * (252 ** 0.5)  # Annualized standard deviation

print('Annualized Average Return:', round(annualized_average_return * 100, 2), '%')
print('Annualized Standard Deviation:', round(annualized_standard_deviation * 100, 2), '%')


# In[6]:


import numpy as np
import yfinance as yf
from scipy.stats import norm

# Step 1: Download historical data for AAPL
ticker = "AAPL"
stock_data = yf.download(ticker, start="2022-04-04", end="2024-04-04")
stock_prices = stock_data['Close']

# Step 2: Define parameters
S0 = stock_prices.iloc[-1]  # Current stock price
K = S0 * 1.1  # Strike price (e.g., 10% above current stock price)
T = 1  # Time to maturity (in years)
r = 0.05  # Risk-free rate
sigma = stock_data['Close'].pct_change().std() * np.sqrt(252)  # Annual volatility
N = 100000  # Number of simulations

# Black-Scholes-Merton method
def black_scholes_merton(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Monte Carlo simulation method
def monte_carlo_simulation(S0, K, T, r, sigma, N):
    z = np.random.standard_normal(N)
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(ST - K, 0)
    call_price = np.exp(-r * T) * np.mean(payoff)
    return call_price

# Calculate option prices using both methods
bsm_price = black_scholes_merton(S0, K, T, r, sigma)
mc_price = monte_carlo_simulation(S0, K, T, r, sigma, N)

# Compare the results
print("Black-Scholes-Merton option price:", bsm_price)
print("Monte Carlo simulation option price:", mc_price)


# In[7]:


import numpy as np
import yfinance as yf

# Step 1: Download historical data for AAPL
ticker = "AAPL"
stock_data = yf.download(ticker, start="2022-04-04", end="2024-04-04")
stock_prices = stock_data['Close']

# Step 2: Define parameters
S0 = stock_prices.iloc[-1]  # Current stock price
K = S0 * 1.1  # Strike price (e.g., 10% above current stock price)
T = 1  # Time to maturity (in years)
r = 0.05  # Risk-free rate
sigma = stock_data['Close'].pct_change().std() * np.sqrt(252)  # Annual volatility
N = 100  # Number of time steps in the binomial tree

# Function to build the binomial tree
def build_binomial_tree(S0, T, r, sigma, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialize stock price tree
    stock_tree = np.zeros((N + 1, N + 1))
    stock_tree[0, 0] = S0

    # Fill the stock price tree
    for i in range(1, N + 1):
        for j in range(i + 1):
            stock_tree[j, i] = stock_tree[0, 0] * (u ** j) * (d ** (i - j))

    return stock_tree

# Function to calculate option price using the binomial tree
def binomial_tree_option_price(S0, K, T, r, sigma, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Build the binomial tree
    stock_tree = build_binomial_tree(S0, T, r, sigma, N)

    # Initialize option price tree
    option_tree = np.zeros((N + 1, N + 1))

    # Calculate option payoff at maturity
    for j in range(N + 1):
        option_tree[j, N] = max(0, stock_tree[j, N] - K)

    # Backward induction to calculate option price at earlier nodes
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_tree[j, i] = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])

    return option_tree[0, 0]

# Calculate option price using the binomial tree method
option_price_binomial_tree = binomial_tree_option_price(S0, K, T, r, sigma, N)

print("Option price using Binomial Tree method:", option_price_binomial_tree)


# In[8]:


import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Download historical data for AAPL
ticker = "AAPL"
stock_data = yf.download(ticker, start="2022-04-04", end="2024-04-04")
stock_prices = stock_data['Close']

# Step 2: Define parameters
S0 = stock_prices.iloc[-1]  # Current stock price
K = S0 * 1.1  # Strike price (e.g., 10% above current stock price)
T = 1  # Time to maturity (in years)
r = 0.05  # Risk-free rate
sigma = stock_data['Close'].pct_change().std() * np.sqrt(252)  # Annual volatility
N = 100  # Number of time steps in the binomial tree

# Function to build the binomial tree
def build_binomial_tree(S0, T, r, sigma, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialize stock price tree
    stock_tree = np.zeros((N + 1, N + 1))
    stock_tree[0, 0] = S0

    # Fill the stock price tree
    for i in range(1, N + 1):
        for j in range(i + 1):
            stock_tree[j, i] = stock_tree[0, 0] * (u ** j) * (d ** (i - j))

    return stock_tree

# Function to calculate option price using the binomial tree
def binomial_tree_option_price(S0, K, T, r, sigma, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Build the binomial tree
    stock_tree = build_binomial_tree(S0, T, r, sigma, N)

    # Initialize option price tree
    option_tree = np.zeros((N + 1, N + 1))

    # Calculate option payoff at maturity
    for j in range(N + 1):
        option_tree[j, N] = max(0, stock_tree[j, N] - K)

    # Backward induction to calculate option price at earlier nodes
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_tree[j, i] = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])

    return option_tree[0, 0]

# Calculate option price using the binomial tree method
option_price_binomial_tree = binomial_tree_option_price(S0, K, T, r, sigma, N)

print("Option price using Binomial Tree method:", option_price_binomial_tree)

# Plotting the option price evolution over time
stock_tree = build_binomial_tree(S0, T, r, sigma, N)
option_prices = [binomial_tree_option_price(price, K, T, r, sigma, N) for price in stock_tree[0]]

plt.plot(stock_tree[0], option_prices)
plt.title("Option Price Evolution with Binomial Tree Method")
plt.xlabel("Stock Price")
plt.ylabel("Option Price")
plt.grid(True)
plt.show()


# In[9]:


import numpy as np
import matplotlib.pyplot as plt

# Function to build the binomial tree
def build_binomial_tree(S0, T, r, sigma, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    
    # Initialize stock price tree
    stock_tree = np.zeros((N + 1, N + 1))
    stock_tree[0, 0] = S0
    
    # Fill the stock price tree
    for i in range(1, N + 1):
        for j in range(i + 1):
            stock_tree[j, i] = S0 * (u ** j) * (d ** (i - j))
    
    return stock_tree

# Define parameters
S0 = 179.48  # Current stock price
T = 1  # Time to maturity (in years)
r = 0.05  # Risk-free rate
sigma = 0.2365  # Annual volatility
N = 100  # Number of time steps in the binomial tree

# Build the binomial tree
stock_tree = build_binomial_tree(S0, T, r, sigma, N)

# Plotting the binomial tree
plt.figure(figsize=(10, 6))
for i in range(N + 1):
    plt.plot(stock_tree[:, i], label=f"Time Step {i}")
plt.title("Binomial Tree for AAPL Stock Prices")
plt.xlabel("Number of Steps")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.show()


# In[10]:


import numpy as np
from scipy.stats import norm

# Option parameters
S0 = 179.48    # Current stock price
K = 197.43     # Strike price
T = 1          # Time to maturity (in years)
r = 0.05       # Risk-free rate
sigma = 0.2365 # Annual volatility
option_price_target = 15.67  # Target option price

# Function to calculate option price using Black-Scholes-Merton formula
def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Function to find implied volatility using Newton-Raphson method
def implied_volatility(S0, K, r, T, option_price, sigma_est=0.2, max_iter=100, tol=1e-5):
    for i in range(max_iter):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma_est**2) * T) / (sigma_est * np.sqrt(T))
        d2 = d1 - sigma_est * np.sqrt(T)
        option_price_est = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        vega = S0 * norm.pdf(d1) * np.sqrt(T)
        error = option_price_est - option_price
        if abs(error) < tol:
            return sigma_est
        sigma_est -= error / vega
    return sigma_est

# Calculate implied volatility
implied_vol = implied_volatility(S0, K, r, T, option_price_target)

# Recalculate option price using the found implied volatility
option_price_actual = black_scholes_call(S0, K, r, implied_vol, T)

# Calculate and print Greeks
def delta(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    return delta

def gamma(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    return gamma

def theta(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    theta = - (S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    return theta / 365  # Convert to daily theta

def vega(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S0 * norm.pdf(d1) * np.sqrt(T) / 100
    return vega

print("Option Price:", option_price_actual)
print("Delta:", delta(S0, K, r, implied_vol, T))
print("Gamma:", gamma(S0, K, r, implied_vol, T))
print("Theta:", theta(S0, K, r, implied_vol, T))
print("Vega:", vega(S0, K, r, implied_vol, T))


# In[11]:


import numpy as np
from scipy.stats import norm

# Option parameters
S0 = 179.48    # Current stock price
K = 197.43     # Strike price
T = 1          # Time to maturity (in years)
r = 0.05       # Risk-free rate
sigma = 0.2365 # Annual volatility
option_price_target = 15.67  # Target option price

# Calculate option price using Black-Scholes-Merton formula
def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Function to find implied volatility using Newton-Raphson method
def implied_volatility(S0, K, r, T, option_price, sigma_est=0.2, max_iter=100, tol=1e-5):
    for i in range(max_iter):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma_est**2) * T) / (sigma_est * np.sqrt(T))
        d2 = d1 - sigma_est * np.sqrt(T)
        option_price_est = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        vega = S0 * norm.pdf(d1) * np.sqrt(T)
        error = option_price_est - option_price
        if abs(error) < tol:
            return sigma_est
        sigma_est -= error / vega
    return sigma_est

# Calculate implied volatility
implied_vol = implied_volatility(S0, K, r, T, option_price_target)

# Recalculate option price using the found implied volatility
option_price_actual = black_scholes_call(S0, K, r, implied_vol, T)

# Calculate Delta
def delta(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    return delta

# Current Delta
current_delta = delta(S0, K, r, implied_vol, T)

# Change in stock price for hedging
S1 = 180.00  # New stock price

# New Delta after change in stock price
new_delta = delta(S1, K, r, implied_vol, T)

# Adjustment in the underlying asset position
position_adjustment = (new_delta - current_delta) * S1

print("Option Price:", option_price_actual)
print("Current Delta:", current_delta)
print("New Delta after stock price change:", new_delta)
print("Position adjustment in the underlying asset:", position_adjustment)


# In[12]:


# Calculate Gamma
def gamma(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    return gamma

# Current Gamma
current_gamma = gamma(S0, K, r, implied_vol, T)

# Change in stock price for hedging
S2 = 180.50  # New stock price

# New Gamma after change in stock price
new_gamma = gamma(S2, K, r, implied_vol, T)

# Adjustment in the underlying asset position
position_adjustment = (new_gamma - current_gamma) * S2

print("Current Gamma:", current_gamma)
print("New Gamma after stock price change:", new_gamma)
print("Position adjustment in the underlying asset:", position_adjustment)


# In[13]:


import numpy as np
import matplotlib.pyplot as plt

# AAPL data
dates = np.array(['2022-04-04', '2022-04-05', '2022-04-06', '2022-04-07', '2022-04-08', '2022-04-09', '2022-04-10'])  # Example dates
delta_values = np.array([0.48585580754172736, 0.4901654969487633, 0.49132918530379866, 0.4898150202251499, 0.49201562136170214, 0.49316134882185424, 0.49426801478960336])  # Example Delta values
gamma_values = np.array([0.008298435663112301, 0.008255888226342977, 0.008213876545292966, 0.008172389740300968, 0.008131416504425794, 0.008090945975220288, 0.008050966782337535])  # Example Gamma values
theta_values = np.array([-0.03603828248311284, -0.03606795090322895, -0.03609759080069596, -0.03612720280392597, -0.036156787531107, -0.03618634559669599, -0.03621587761181667])  # Example Theta values
vega_values = np.array([0.7155715371946172, 0.716, 0.716, 0.716, 0.716, 0.716, 0.716])  # Example Vega values

# Option parameters
S0 = 179.48    # Current stock price
K_call = 180   # Call option strike price
K_put = 175    # Put option strike price
T = 1          # Time to maturity (in years)
r = 0.05       # Risk-free rate
sigma = 0.2365 # Annual volatility

# Plot Delta for call option
plt.figure(figsize=(10, 6))
plt.plot(dates, delta_values, marker='o', color='b', label='Delta (Call)')
plt.title('Delta Over Time for Call Option')
plt.xlabel('Date')
plt.ylabel('Delta Value')
plt.xticks(rotation=45)
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Delta for put option
plt.figure(figsize=(10, 6))
plt.plot(dates, delta_values, marker='o', color='r', label='Delta (Put)')
plt.title('Delta Over Time for Put Option')
plt.xlabel('Date')
plt.ylabel('Delta Value')
plt.xticks(rotation=45)
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Gamma for call option
plt.figure(figsize=(10, 6))
plt.plot(dates, gamma_values, marker='o', color='g', label='Gamma (Call)')
plt.title('Gamma Over Time for Call Option')
plt.xlabel('Date')
plt.ylabel('Gamma Value')
plt.xticks(rotation=45)
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Gamma for put option
plt.figure(figsize=(10, 6))
plt.plot(dates, gamma_values, marker='o', color='orange', label='Gamma (Put)')
plt.title('Gamma Over Time for Put Option')
plt.xlabel('Date')
plt.ylabel('Gamma Value')
plt.xticks(rotation=45)
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Theta for call option
plt.figure(figsize=(10, 6))
plt.plot(dates, theta_values, marker='o', color='purple', label='Theta (Call)')
plt.title('Theta Over Time for Call Option')
plt.xlabel('Date')
plt.ylabel('Theta Value')
plt.xticks(rotation=45)
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Theta for put option
plt.figure(figsize=(10, 6))
plt.plot(dates, theta_values, marker='o', color='brown', label='Theta (Put)')
plt.title('Theta Over Time for Put Option')
plt.xlabel('Date')
plt.ylabel('Theta Value')
plt.xticks(rotation=45)
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Vega for call option
plt.figure(figsize=(10, 6))
plt.plot(dates, vega_values, marker='o', color='cyan', label='Vega (Call)')
plt.title('Vega Over Time for Call Option')
plt.xlabel('Date')
plt.ylabel('Vega Value')
plt.xticks(rotation=45)
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Vega for put option
plt.figure(figsize=(10, 6))
plt.plot(dates, vega_values, marker='o', color='pink', label='Vega (Put)')
plt.title('Vega Over Time for Put Option')
plt.xlabel('Date')
plt.ylabel('Vega Value')
plt.xticks(rotation=45)
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




