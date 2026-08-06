import numpy as np
import matplotlib.pyplot as plt

# 1. Define Parameters (Simulation Inputs)
# S0: Initial stock price ($)
# mu: Expected annual return (Drift) -> 10%
# sigma: Annualized volatility (Diffusion/Risk) -> 20%
# T: Time horizon (1 year)
# dt: Time step (1 trading day = 1/252)
# N: Total number of time steps
# Simulations: Number of generated price paths
S0 = 100
mu = 0.10
sigma = 0.20
T = 1.0
dt = 1/252
N = 252
Simulations = 1000

# 2. Random Number Generator
# Generate a matrix (1000 x 252) of random numbers from a standard normal distribution.
# This represents the Wiener Process (Brownian Motion) component 'dW'.
Z = np.random.normal(0, 1, (Simulations, N))

# 3. Geometric Brownian Motion (GBM) Formula
# S_t = S_{t-1} * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
# Initialize price matrix
S = np.zeros((Simulations, N + 1))
S[:, 0] = S0

# Simulate price paths day by day
for t in range(1, N + 1):
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z[:, t-1]
    S[:, t] = S[:, t-1] * np.exp(drift + diffusion)

# 4. Visualization:
plt.figure(figsize=(10, 6))
plt.plot(S.T, alpha=0.1, color='#1f77b4') # Set transparency to see density
plt.plot(S.T[:, 0], color='red', linewidth=2, label='Initial Price')

plt.title(f'Monte Carlo Simulation: {Simulations} Scenarios based on GBM')
plt.xlabel('Trading Days')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 5. Risk Analysis: Value at Risk (VaR)
# Analyzing the distribution of final prices to estimate downside risk.
final_prices = S[:, -1]
VaR_95 = np.percentile(final_prices, 5) # The 5th percentile (95% confidence level)

print(f"--- Simulation Results ---")
print(f"Initial Price: ${S0}")
print(f"Expected Price (Mean): ${np.mean(final_prices):.2f}")
print(f"Value at Risk (95% Confidence): ${S0 - VaR_95:.2f}")
print(f"(Meaning: There is a 5% chance the loss will exceed ${S0 - VaR_95:.2f})")
