#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 20:56:57 2021

@author: apple
"""

import numpy as np
import cvxpy as cp
import pandas as pd


# Markowitz Mean Variance 
def markowitz_Portfolio(lam,mu, sigma):
    n = np.shape(sigma)[0]
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, sigma) - lam * mu.T @ w),
                     [w >= np.zeros(n),
                     cp.sum(w) == 1])
    solve = prob.solve()
    return w.value

def Portfolio_Expected_Return(lam,mu,sigma):
    w = markowitz_Portfolio(lam,mu,sigma)
    expectedReturn = np.dot(w,mu)
    return expectedReturn

def Portfolio_Volatility(lam,mu,sigma):
    w = markowitz_Portfolio(lam,mu,sigma)
    vol = np.sqrt(w.T @ sigma @ w)
    return vol

#Uniform
def EWP(N):
    return np.ones(N)/N

#Global minimum variance portfolio (GMVP)
def GMVP(sigma,mu):
    w = cp.Variable(len(mu))
    var = cp.quad_form(w,sigma)
    problem = cp.Problem(cp.Minimize(var), [w >= 0, cp.sum(w) == 1])
    problem.solve()
    return w.value

#Maximum Sharpe Ratio Portfolio (MSRP) Using Schaible
def MSRP(sigma,mu):
    w = cp.Variable(len(mu))
    var = cp.quad_form(w,sigma)
    problem = cp.Problem(cp.Minimize(var), [w >= 0 , cp.sum(w * mu) == 1])
    problem.solve()
    quo = w.value/np.sum(w.value)
    return quo

#Inverse volatility portfolio (IVP)
def IVP(sigma):
    varSigma = np.sqrt(np.diag(sigma))
    w = 1/varSigma
    w = w/np.sum(w)
    return w

#Maximum expected return portfolio (MRP)
def MRP(mu):
    w = cp.Variable(len(mu))
    prob = cp.Problem(cp.Maximize(w.T @ mu), [w >= np.zeros(len(mu)),cp.sum(w) == 1])
    _ = prob.solve()
    return w.value

#Mean Downside Risk (DR), vary alpha = 1,2,3
def portfolioDR(X, lmd = 0.5, alpha = 2):
    T = len(X)
    mu = np.mean(X).to_numpy()
    w = cp.Variable(len(X.columns))
    X = X.to_numpy()
    prob = cp.Problem(cp.Maximize(w.T @ mu - (lmd/T) * cp.pnorm(mu.T @ w - X @ w,alpha)), 
                      [w >= 0, sum(w)==1])
    prob.solve()
    return w.value

# Mean CVar Portfolio
def portfolioCVaR(X, lmd = 0.5, alpha = 0.95):
    T = len(X)
    mu = np.mean(X).to_numpy()
    
    #Variables
    w = cp.Variable(len(X.columns))
    z = cp.Variable(T)
    zeta = cp.Variable(1)
    
    X = X.to_numpy()

    prob = cp.Problem(cp.Maximize(w.T @ mu - lmd*zeta - (lmd/(T*(1-alpha))) * sum(z)),
                     [z >= 0, z >= -X @ w - zeta, w >= 0, sum(w) == 1])
    prob.solve()
    return w.value

# Portfolio Max DD
def portfolioMaxDD(X, data, c= 0.2):
    T = len(X)
    N = len(X.columns)
    mu = np.mean(X).to_numpy()
    X = X.to_numpy()
    
    dfCumsum = pd.DataFrame(columns = data.columns)
    for n in data.columns:
        dfCumsum[n] = data[n].cumsum()
    X_cum = dfCumsum.iloc[-1].to_numpy()

    #Variables
    w = cp.Variable(N)
    u = cp.Variable(T)
    
    #Problem
    
    prob = cp.Problem(cp.Maximize(w.T @ mu),
                     [w >= 0, sum(w) == 1,
                     u <= X_cum @ w + c,
                     u >= X_cum @ w,
                     #u[-1] >= u[-T]
                     u[1:] >= u[:T-1]])
    prob.solve()
    return w.value

# Ave DD
def portfolioAveDD(X, data ,c = 0.2):
    T = len(X)
    N = len(X.columns)
    mu = np.mean(X).to_numpy()
    X = X.to_numpy()
    
    dfCumsum = pd.DataFrame(columns = data.columns)
    for n in data.columns:
        dfCumsum[n] = data[n].cumsum()
    X_cum = dfCumsum.iloc[-1].to_numpy()

    #Variables
    w = cp.Variable(N)
    u = cp.Variable(T)
    
    prob = cp.Problem(cp.Maximize(w.T @ mu),
                     [w >= 0, sum(w) == 1,
                     np.mean(u) <= np.mean(X_cum @ w) + c,
                     u >= X_cum @ w,
#                      u[-1] >= u[-T]])
                     u[1:] >= u[:T-1]])
    prob.solve()
    return w.value

# Portfolio Max Return Robust Elliposid
def portfolioMaxReturnRobustEllipsoid(mu_hat, S, kappa = 0.1):
    S12 = np.linalg.cholesky(S)
    N = len(mu_hat)
    w = cp.Variable(N)
    var = S12 @ w
    prob = cp.Problem(cp.Maximize(w.T @ mu_hat - kappa * cp.norm(S12 @ w,2)),
                    [w >= 0, sum(w) == 1])
    prob.solve()
    return w.value

# Portfolio Markowitz Robust
def portfolioMarkowitzRobust(mu_hat, sigma_hat, kappa, delta_, lmd = 0.5):
    N = len(mu_hat)
    S12 = np.linalg.cholesky(sigma_hat)
    w = cp.Variable(N)
    prob = cp.Problem(cp.Maximize(w.T @ mu_hat - kappa * cp.norm(S12 @ w, 2) - 
                                  lmd*(cp.norm(S12 @ w,2) + delta_ * cp.norm(w))**2),
                     [w >= 0, sum(w) == 1])
    prob.solve()
    return w.value


