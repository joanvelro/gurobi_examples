#!/usr/bin/env python3.7

"""
Copyright 2021, Gurobi Optimization, LLC

Portfolio selection: given a sum of money to invest, one must decide how to
spend it amongst a portfolio of financial securities.  Our approach is due
to Markowitz (1959) and looks to minimize the risk associated with the
investment while realizing a target expected return.  By varying the target,
one can compute an 'efficient frontier', which defines the optimal portfolio
for a given expected return.

Note that this example reads historical return data from a comma-separated
 file (../data/portfolio.csv).  As a result, it must be run from the Gurobi
 examples/python directory.

 This example requires the pandas (>= 0.20.3), NumPy, and Matplotlib
 Python packages, which are part of the SciPy ecosystem for
 mathematics, science, and engineering (http://scipy.org).  These
 packages aren't included in all Python distributions, but are
 included by default with Anaconda Python.
"""

import gurobipy as gp
from gurobipy import GRB
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import (normalized) historical return data using pandas
data = pd.read_csv('../data/portfolio.csv', index_col=0)
stocks = data.columns

###############
# PARAMETERS
#############
# Standard deviation per stock
stock_volatility = data.std()
# Expected value per stock
stock_return = data.mean()
# Covaraince matrix stocks
sigma = data.cov()

###############
# MODEL
#############

# Create an empty model
m = gp.Model('portfolio')

# ---------
# VARIABLES
# ----------
# Participation per stock
vars = pd.Series(m.addVars(stocks, lb=0.0, ub=5), index=stocks)



# ---------
# OBJECTIVE
# ----------
# Objective is to minimize risk (squared).  This is modeled using the
# covariance matrix, which measures the historical correlation between stocks.
portfolio_risk = sigma.dot(vars).dot(vars)  # Variance X S X^T
m.setObjective(portfolio_risk, gp.GRB.MINIMIZE)

# -----------
# CONSTRAINTS
# -----------

#
# Constraint (1)
#
# Fix budget with a constraint
# Unitary budget
m.addConstr(vars.sum() == 1, 'budget')
#m.addConstr(sum(vars[i]*stock_return[i] for i in stocks) == 1, 'budget')



# Optimize model to find the minimum risk portfolio
m.setParam('OutputFlag', 1)
m.optimize()

# -----------
# EXPRESIONS
# -----------
# Create an expression representing the expected return for the portfolio
portfolio_return = stock_return.dot(vars)

# Display minimum risk portfolio
print('\n')
print(' --- Minimum Risk Portfolio --- \n')
for v in vars:
    if v.x > 0:
        #print('\t%s\t: %g' % (v.varname, v.x))
        print('{} - Stock {} : {}'.format(v.varname, stocks[v.index] , v.x ))
minrisk_volatility = sqrt(portfolio_risk.getValue())
print('\nVolatility      = %g' % minrisk_volatility)
minrisk_return = portfolio_return.getValue()
print('Expected Return = %g' % minrisk_return)

# Add (redundant) target return constraint
target = m.addConstr(portfolio_return == minrisk_return, 'target')

# Solve for efficient frontier by varying target return
frontier = pd.Series(dtype=np.float64)
for r in np.linspace(stock_return.min(), stock_return.max(), 100):
    target.rhs = r
    m.optimize()
    frontier.loc[sqrt(portfolio_risk.getValue())] = r


# =================
# PLOTS
# ================

def plot_portfolio(stock_volatility, minrisk_volatility, minrisk_return, stock_return, frontier):
    # Plot volatility versus expected return for individual stocks
    fig = plt.figure(figsize=(12, 6), dpi=100, facecolor='w', edgecolor='w')
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(x=stock_volatility,
               y=stock_return,
               color='Blue',
               label='Individual Stocks')
    for i, stock in enumerate(stocks):
        ax.annotate(stock, (stock_volatility[i], stock_return[i]))

    # Plot volatility versus expected return for minimum risk portfolio
    ax.scatter(x=minrisk_volatility,
               y=minrisk_return,
               color='DarkGreen',
               label='Minimum Risk Point')
    ax.annotate('Minimum\nRisk\nPortfolio',
                (minrisk_volatility, minrisk_return),
                horizontalalignment='right')

    # Plot efficient frontier
    ax.plot(frontier,
            color='DarkGreen',
            label='Efficient Frontier')

    ax.hlines(y=np.zeros(len(np.arange(0, max(stock_volatility), 0.005))),
              xmin=0,
              xmax=max(stock_volatility),
              color='gray',
              linestyle='dashed',
              linewidth=1)
    # Format and display the final plot
    ax.axis([0.005, 0.06, -0.02, 0.025])
    ax.set_xlabel('Volatility (standard deviation)')
    ax.set_ylabel('Expected Return')
    ax.legend()
    ax.grid()
    plt.savefig('../figures/portfolio.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
    # print("Plotted efficient frontier to 'portfolio.png'")


def plot_stocks_timeseries(data):
    fig = plt.figure(figsize=(16, 6), dpi=120, facecolor='w', edgecolor='w')

    for i in np.arange(0, data.shape[1]):
        ax = fig.add_subplot(5, 2, i + 1)
        ax.plot(data[data.columns[i]],
                label=data.columns[i],
                linestyle='dashed', color='k', linewidth=1)

        # ax.axis([0.005, 0.06, -0.02, 0.025])
        ax.set_xlabel('Time')
        ax.set_ylabel('Stock Value')
        # ax.set_title(data.columns[i])
        ax.legend()
        ax.grid()
    plt.subplots_adjust(wspace=0.3, hspace=1.2)
    plt.savefig('../figures/portfolio_stocks.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()


plot_portfolio(stock_volatility, minrisk_volatility, minrisk_return, stock_return, frontier)
plot_stocks_timeseries(data)
