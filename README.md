# Portfolio Optimizer

A Python-based portfolio optimization tool that implements Modern Portfolio Theory to find the optimal asset allocation based on the Sharpe Ratio and Efficient Frontier analysis.

## Features

- **Sharpe Ratio Optimization**: Automatically finds portfolio weights that maximize the Sharpe Ratio
- **Efficient Frontier Analysis**: Visualizes the risk-return tradeoff and identifies optimal portfolios
- **Monte Carlo Simulation**: Generates random portfolio weights to explore the feasible investment space
- **Flexible Asset Allocation**: Supports custom minimum and maximum allocation constraints for each asset
- **Risk-Free Rate**: Customizable risk-free rate for Sharpe Ratio calculations
- **Historical Data**: Uses `yfinance` to fetch historical price data automatically

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install numpy pandas scipy matplotlib yfinance
```

## Usage

### Basic Usage

```python
from portfolio_optimizer import PortfolioOptimizer

# Initialize with a list of tickers
tickers = ['BTC-GBP', 'V3AB.L', 'VFEG.L', 'CNX1.L', '0P0000TKZO.L', 'PHAU.L']
optimizer = PortfolioOptimizer(tickers, start='2020-1-1')

# Get the optimal portfolio based on Sharpe Ratio
optimal_portfolio = optimizer.optimal_sharpe_portfolio
print(optimal_portfolio)
```

### Visualizations

```python
# Generate Monte Carlo simulation plot
optimizer.mcs_port_diagram(I=100000)

# Generate Efficient Frontier plot
optimizer.efficient_frontier(I=100000)
```

### Custom Portfolio Analysis

```python
# Get portfolio for a target volatility
portfolio = optimizer.portfolio_for_volatility(vol=0.15)

# Get portfolio for a target return
portfolio = optimizer.portfolio_for_returns(ret=0.10)
```

## Features in Detail

### Optimal Sharpe Portfolio
Returns a dictionary containing:
- Expected annual return
- Expected annual volatility
- Sharpe Ratio
- Optimal weights for each asset

### Monte Carlo Simulation
- Generates random portfolio weights within allocation constraints
- Plots expected returns vs volatility
- Color-codes portfolios based on Sharpe Ratio

### Efficient Frontier
- Plots the efficient frontier curve
- Shows the optimal Sharpe Ratio portfolio point
- Returns portfolios that maximize return for given risk levels

## Implementation Details

### Risk and Return Calculations
- Returns are calculated using logarithmic returns
- Portfolio volatility uses the full covariance matrix
- Supports constraints on individual asset allocations

### Optimization Methods
- Uses `scipy.optimize` for Sharpe Ratio maximization
- Implements vectorized operations for Monte Carlo simulations
- Supports constrained optimization for the Efficient Frontier

## Future Improvements

- Interactive plotting interface
- Additional portfolio optimization metrics
- Implementation of more realistic market behavior models:
  - Jump diffusion
  - Stochastic volatility

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
