import yfinance as yf
import scipy.optimize as sco
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


class PortfolioOptimizer():

    def __init__(self, tickers, start=None, min_alloc=0., max_alloc=1., r=0.):

        if start is None:
            start = (pd.Timestamp.now() - pd.DateOffset(years=2)).strftime('%Y-%m-%d')

        self.data = yf.download(tickers, start=start)['Adj Close'].dropna()
        self.asset_names = list(self.data.columns)
        self.rets = np.log(self.data / self.data.shift(1))
        self.num_of_assets = len(self.rets.columns)

        self.min_alloc = min_alloc
        self.max_alloc = max_alloc

        self.r = r
        self.optimize_sharpe()
        self.opt_sharpe_ratio = -self.opt_sharpe.fun
        self.opt_sharpe_weight = self.opt_sharpe.x

        self.t_rets = None
        self.t_vols = None
        self.t_weights = None

    def optimize_sharpe(self):

        def min_sharpe(weights):
            return - (self.port_rets(weights) - self.r) / self.port_vols(weights)
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((self.min_alloc, self.max_alloc) for x in range(self.num_of_assets))
        eweights = np.array(self.num_of_assets * (1. / self.num_of_assets,))

        self.opt_sharpe = sco.minimize(min_sharpe, eweights, method='SLSQP', bounds=bnds, constraints=cons)

    @property
    def optimal_sharpe_portfolio(self):
        return {
            'return': round(float(self.port_rets(self.opt_sharpe_weight)), 3),
            'volatility': round(float(self.port_vols(self.opt_sharpe_weight)), 3),
            'sharpe_ratio': round(float(self.opt_sharpe_ratio), 3),
            'weights': dict(zip(self.asset_names, [round(float(w), 3) for w in self.opt_sharpe_weight]))
        }

    def port_rets(self, weights):
        return np.sum(self.rets.mean() * np.array(weights)) * 252

    def port_vols(self, weights):
        return np.sqrt(np.dot(np.array(weights).T, np.dot(self.rets.cov() * 252, np.array(weights))))

    def port_sharpe(self, weights):
        return (self.port_rets(weights) - self.r) / self.port_vols(weights)
    
    def generate_constrained_weights(self, num_of_assets, min_alloc, max_alloc):
        """
        Efficiently generate random weights that satisfy allocation constraints
        using the stick-breaking method with rejection sampling only on valid ranges.
        """
        weights = np.zeros(num_of_assets)
        remaining = 1.0
        remaining_assets = num_of_assets
        
        for i in range(num_of_assets - 1):
            # Calculate valid range for this asset
            min_for_this = max(min_alloc, 
                            remaining - (remaining_assets - 1) * max_alloc)
            max_for_this = min(max_alloc, 
                            remaining - (remaining_assets - 1) * min_alloc)
            
            # Generate a valid weight for this asset
            if min_for_this <= max_for_this:
                weights[i] = np.random.uniform(min_for_this, max_for_this)
            else:
                # If no valid range, distribute remaining equally
                return np.full(num_of_assets, 1.0/num_of_assets)
            
            remaining -= weights[i]
            remaining_assets -= 1
        
        # Set the last weight
        weights[-1] = remaining
        
        # Verify constraints are met, if not return equal weights
        if (weights[-1] < min_alloc or weights[-1] > max_alloc or 
            not np.isclose(np.sum(weights), 1.0)):
            return np.full(num_of_assets, 1.0/num_of_assets)
        
        return weights

    def mcs_port_diagram(self, I=10000, plot=True):

        p_rets = []
        p_vols = []
        
        for p in range(I):
        
            # generate and normalize weights
            weights = self.generate_constrained_weights(
                self.num_of_assets,
                self.min_alloc,
                self.max_alloc
            )
            
            # store each weight and volatility
            p_rets.append(self.port_rets(weights))
            p_vols.append(self.port_vols(weights))
        
        # convert into np.array for visualization
        p_rets = np.array(p_rets)
        p_vols = np.array(p_vols)

        # create a scatter plot with a Sharpe ratio heatmap
        if plot:
            plt.scatter(
                p_vols, p_rets,
                c=((p_rets - self.r) / p_vols),
                marker='o', cmap='coolwarm'
            )
            plt.xlabel('expected volatility')
            plt.ylabel('expected return')
            plt.colorbar(label='Sharpe ratio')
            plt.show()

        return min(p_vols), max(p_vols), p_vols, p_rets  


    def efficient_frontier(self, plot=True, I=10000):

        eweights = np.array(self.num_of_assets * (1. / self.num_of_assets,))

        def min_rets(weights):
            return - self.port_rets(weights)

        cons = ({'type': 'eq', 'fun': lambda x: self.port_vols(x) - t_vol},
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1 })
        
        bnds = tuple((self.min_alloc, self.max_alloc) for x in range(self.num_of_assets))

        low_vol, high_vol, scatter_vols, scatter_rets = self.mcs_port_diagram(I=I, plot=False)

        t_vols = np.linspace(low_vol, high_vol, 50)
        t_rets = []
        weights = []
        for t_vol in t_vols:
            res = sco.minimize(min_rets, eweights, method='SLSQP',
                            bounds=bnds, constraints=cons)
            t_rets.append(-res['fun'])
            weights.append(res.x)

        t_rets = np.array(t_rets)

        if plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(scatter_vols, scatter_rets,
                        c=(scatter_rets / scatter_vols),
                        marker='o', cmap='coolwarm')
            plt.xlabel('expected volatility')
            plt.ylabel('expected return')
            plt.colorbar(label='Sharpe ratio')

            plt.plot(t_vols, t_rets, lw=2.0)

            plt.plot(self.port_vols(self.opt_sharpe_weight),
                    self.port_rets(self.opt_sharpe_weight),
                    'b*', markersize=15)
            plt.show()
            
        self.t_vols, self.t_rets, self.t_weights = t_vols, t_rets, weights

    def find_nearest(self, array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
            return idx - 1
        else:
            return idx

    def portfolio_for_volatility(self, vol):
        if self.t_vols is None:
            self.efficient_frontier(plot=False)

        idx = self.find_nearest(self.t_vols, vol)
        returns = self.t_rets[idx]
        weights = dict(zip(self.asset_names, [round(float(w), 3) for w in self.t_weights[idx]]))

        return {'returns': round(float(returns), 3), 'weights': weights}
    
    def portfolio_for_returns(self, ret):
        if self.t_rets is None:
            self.efficient_frontier(plot=False)

        idx = self.find_nearest(self.t_rets, ret)
        volatility = self.t_vols[idx]
        weights = dict(zip(self.asset_names, [round(float(w), 3) for w in self.t_weights[idx]]))

        return {'volatility': round(float(volatility), 3), 'weights': weights}



tickers = ['AAPL', 'TSLA', 'MSFT']
optimizer = PortfolioOptimizer(tickers, max_alloc=0.5)

print(optimizer.optimal_sharpe_portfolio)

optimizer.efficient_frontier()