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
        weights = np.array(weights)
        mean_returns = np.array(self.rets.mean())

        if weights.ndim == 1:
            return float(np.sum(self.rets.mean() * np.array(weights)) * 252)
        else:
            return np.sum(mean_returns[np.newaxis, :] * weights, axis=1) * 252
    
    def port_vols(self, weights):
        weights = np.array(weights)
        cov_matrix = self.rets.cov() * 252

        if weights.ndim == 1:
            return float(np.sqrt(np.sum(weights * (weights @ cov_matrix))))
        else:
            return np.sqrt(np.sum(weights * (weights @ cov_matrix), axis=1))

    def port_sharpe(self, weights):
        return float((self.port_rets(weights) - self.r) / self.port_vols(weights))
    
    def generate_constrained_weights(self, I):
        """Generate I sets of weights at once"""
        weights = np.zeros((I, self.num_of_assets))
        remaining = np.ones(I)
        
        for i in range(self.num_of_assets - 1):
            # Calculate valid ranges for all simulations at once
            min_for_this = np.maximum(
                self.min_alloc,
                remaining - (self.num_of_assets - i - 1) * self.max_alloc
            )
            max_for_this = np.minimum(
                self.max_alloc,
                remaining - (self.num_of_assets - i - 1) * self.min_alloc
            )
            
            # Generate weights for this asset for all simulations
            weights[:, i] = np.random.uniform(
                min_for_this, 
                max_for_this
            )
            remaining -= weights[:, i]
        
        # Set final weights
        weights[:, -1] = remaining
        
        # Return equal weights for any invalid combinations
        invalid_mask = (
            (weights < self.min_alloc).any(axis=1) | 
            (weights > self.max_alloc).any(axis=1) |
            ~np.isclose(weights.sum(axis=1), 1.0)
        )
        weights[invalid_mask] = np.full(self.num_of_assets, 1.0/self.num_of_assets)

        return weights

    def mcs_port_diagram(self, I=10000, plot=True):
        weights = self.generate_constrained_weights(I)
        p_rets = self.port_rets(weights)
        p_vols = self.port_vols(weights)

        if plot:
                plt.scatter(
                    p_vols, p_rets,
                    c=((p_rets) / p_vols),
                    marker='o', cmap='coolwarm'
                )
                plt.xlabel('expected volatility')
                plt.ylabel('expected return')
                plt.colorbar(label='Sharpe ratio')
                plt.show()

        return p_vols, p_rets  


    def efficient_frontier(self, plot=True, I=10000):

        eweights = np.array(self.num_of_assets * (1. / self.num_of_assets,))

        def min_rets(weights):
            return - self.port_rets(weights)

        cons = ({'type': 'eq', 'fun': lambda x: self.port_vols(x) - t_vol},
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1 })
        
        bnds = tuple((self.min_alloc, self.max_alloc) for x in range(self.num_of_assets))

        scatter_vols, scatter_rets = self.mcs_port_diagram(I=I, plot=False)

        t_vols = np.linspace(min(scatter_vols), max(scatter_vols), 50)
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


# tickers = ['BTC-GBP', 'V3AB.L', 'VFEG.L' ,'CNX1.L', '0P0000TKZO.L', 'PHAU.L']
# optimizer = PortfolioOptimizer(tickers, start='2020-1-1')

# print(optimizer.optimal_sharpe_portfolio)

# optimizer.efficient_frontier()
