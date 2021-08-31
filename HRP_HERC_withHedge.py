#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 21:44:22 2021

@author: apple
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from gap_statistic import OptimalK

import fastcluster
from scipy.cluster.hierarchy import fcluster

from backtest import df_to_matrix,indexCovMatrix

from scipy.stats import t
from arch import arch_model
from statsmodels.tsa.arima_model import ARIMA
import scipy.stats as stats

import pmdarima as pm
import statsmodels.api as sm
import sys

#HRP
#The type of dataframe that should be going in is a dataframe with companies index incrementally
# 1,2,3 etc

def seriation(tree, points, index):

    if index < points:
        return [index]
    
    else:
        
        left = int(tree[index - points, 0])
        right = int(tree[index - points, 1])
        return (seriation(tree, points, left) + seriation(tree, points, right))

    
def compute_serial_matrix(distanceMatrix, method="ward"):

    num = len(distanceMatrix)
    
    flatDistMat = squareform(distanceMatrix)
    
    resLinkage = linkage(flatDistMat, method=method)
    resOrder = seriation(resLinkage, num, num + num - 2)
    
    
    seriatedDist = np.zeros((num, num))
    x,y = np.triu_indices(num, k=1)
    
    
    seriatedDist[x,y] = distanceMatrix[[resOrder[i] for i in x], [resOrder[j] for j in y]]
    seriatedDist[x,y] = seriatedDist[x,y]
    
    return seriatedDist, resOrder, resLinkage


def compute_HRP_weights(covar, resOrder):
    weights = pd.Series(1, index=resOrder)
    alphas = [resOrder]

    while len(alphas) > 0:
        alphas = [cluster[start:end] for cluster in alphas
                            for start, end in ((0, len(cluster) // 2),
                                               (len(cluster) // 2, len(cluster)))
                            if len(cluster) > 1]
        for subcluster in range(0, len(alphas), 2):
            lc = alphas[subcluster]
            
            #Left Side
            leftCovar = covar[lc].loc[lc]
            inv_diag = 1 / np.diag(leftCovar.values)
            parity_w = inv_diag * (1 / np.sum(inv_diag))
            leftVar = np.dot(parity_w, np.dot(leftCovar, parity_w))
            
            #Right Side            
            rc = alphas[subcluster + 1]
            rightCovar = covar[rc].loc[rc]
            inv_diag = 1 / np.diag(rightCovar.values)
            parity_w = inv_diag * (1 / np.sum(inv_diag))
            rightVar = np.dot(parity_w, np.dot(rightCovar, parity_w))

            alloc_factor = 1 - leftVar / (leftVar + rightVar)

            weights[lc] *= alloc_factor
            weights[rc] *= 1 - alloc_factor
            
    return weights

#Dataframe of returns
def HRP(df):
    estimateCor = df.corr(method='pearson')
    estimateCov, column_dic = indexCovMatrix(df)
    # estimate_covar, column_dic = indexCorrMatrix(df.cov())
    distances = np.sqrt((1 - estimateCor) / 2)
    
    orderedDistanceMatrix, resOrder, linkageType = compute_serial_matrix(distances.values, method='single')
    
    HRP_w = compute_HRP_weights(estimateCov, resOrder)
    
    dictOrder = dict(map(reversed, column_dic.items()))
    
    HRP_w = HRP_w.rename(index = dictOrder)
    
    return HRP_w


#HERC
def intersection(list1, list2): 
    intersec = [set(list1) & set(list2)]
    return intersec


def compute_allocation(covar, clusters,Z,dimensions):
    numClusters = len(clusters)
    aWeights = np.array([1.] * len(covar))
    
    cWeights = np.array([1.] * numClusters)
    cVar = np.array([0.] * numClusters)
    
    for i, cluster in clusters.items():
        cluster_covar = covar[cluster, :][:, cluster]
        inv_diag = 1 / np.diag(cluster_covar)
        aWeights[cluster] = inv_diag / np.sum(inv_diag)
        
    for i, cluster in clusters.items():
        weights = aWeights[cluster]
        cVar[i - 1] = np.dot(
            weights, np.dot(covar[cluster, :][:, cluster], weights))
        
    for m in range(numClusters - 1):
        left = int(Z[dimensions - 2 - m, 0])
        lc = seriation(Z, dimensions, left)
        
        right = int(Z[dimensions - 2 - m, 1])
        rc = seriation(Z, dimensions, right)


        id_lc = []
        id_rc = []
        
        for i, cluster in clusters.items():
            if sorted(intersection(lc, cluster)) == sorted(cluster):
                id_lc.append(i)
            if sorted(intersection(rc, cluster)) == sorted(cluster):
                id_rc.append(i)


        id_lc = np.array(id_lc) - 1
        id_rc = np.array(id_rc) - 1

        alpha = 0
        lcVar = np.sum(cVar[id_lc])
        rcVar = np.sum(cVar[id_rc])
        alpha = lcVar / (lcVar + rcVar)

        cWeights[id_lc] = cWeights[
            id_lc] * alpha
        cWeights[id_rc] = cWeights[
            id_rc] * (1 - alpha)

    for i, cluster in clusters.items():
        aWeights[cluster] = aWeights[cluster] * cWeights[
            i - 1]
        
    return aWeights

#Dataframe of returns
def HERC(mat_ret):
    #Need to first calculate the optimal number of clusters
    #The mat_ret that goes into this must be a np array of returns
    # correl_mat = mat_ret.corr(method='pearson')
    column_dic = {k:v for v, k in enumerate(mat_ret.columns)}
    
    correl_mat = df_to_matrix(mat_ret.corr(method='pearson'))
    dist = 1 - correl_mat
    dim = len(dist)
    
    tri_a, tri_b = np.triu_indices(dim, k = 1)
    
    
    
    Z = fastcluster.linkage(dist[tri_a, tri_b], method='ward')
    
    optimalK = OptimalK(parallel_backend = 'rust')
    n_clusters = optimalK(mat_ret.values, cluster_array = np.arange(1,len(mat_ret)))
   
    nb_clusters = n_clusters
    clustering_inds = fcluster(Z, nb_clusters, criterion='maxclust')
    clusters = {i: [] for i in range(min(clustering_inds),max(clustering_inds) + 1)}
    for i, v in enumerate(clustering_inds):
        clusters[v].append(i)
    
    HERC_w = compute_allocation(correl_mat, clusters, Z, dim)
    HERC_w = pd.Series(HERC_w)
    my_inverted_dict = dict(map(reversed, column_dic.items()))
    
    HERC_w = HERC_w.rename(index = my_inverted_dict)
    
    return HERC_w
    
        
#appl = dataReturn(['V','M','X','F','T'],'2020-01-01','2020-12-31', True)


# weights_HERC = HERC(appl)
#weights_HRP = HRP(appl)

#Equity Allocation
#List of Returns
def t_dist(returns, number):
    degrees_of_freedom = len(returns) - 1
    newReturn = t.rvs(degrees_of_freedom, size = number)
    return newReturn


def sd_estimation(returns):
    am = arch_model(returns, p=1,q=1,o=1,dist = "StudentsT", vol = 'EGARCH')
    res = am.fit()
    lastVol = res.conditional_volatility[-1]
    return lastVol, res.params.omega, res.params['alpha[1]'], res.params['beta[1]'], res.params['gamma[1]']

def sd_estimation2(returns):
    am = arch_model(returns, p=1,q=1,dist = "StudentsT", vol = 'GARCH')
    res = am.fit()
    lastVol = res.conditional_volatility[-1]
    return lastVol, res.params.omega, res.params['alpha[1]'], res.params['beta[1]'], res.resid[-1]


def sd_estimation_equation(returns, numberofData):
    lastVol, omega, alpha, beta, gamma  = sd_estimation(returns)
    tDistReturns = t_dist(returns, numberofData)
    prev = 0
    n = 0
    arr = []
    while n < numberofData:
        if n == 0:
            prev = lastVol
        sigma = np.sqrt(np.exp(prev)) * tDistReturns[n]
        newVal = omega + alpha * ((np.abs(sigma) + gamma * sigma)/ np.sqrt(np.exp(prev))) + beta * prev
        print(newVal)
        prev = newVal
        # arr.append(np.sqrt(prev))
        arr.append(np.sqrt(np.exp(prev)))
        n = n + 1
    return arr

def sd_estimation_equation2(returns, numberofData):
    am = arch_model(returns, p=1,q=1,dist = "StudentsT", vol = 'GARCH')
    res = am.fit()
    arr = res.forecast(horizon = numberofData).variance[-1:].values[-1]
    
    # lastVol, omega, alpha, beta, resid = sd_estimation2(returns)
    # tDistReturns = t_dist(returns, numberofData)
    # prev = 0
    # n = 0
    # arr = []
    # while n < numberofData:
    #     if n == 0:
    #         # prev = lastVol
    #         prev = 0.01 * np.sqrt(omega + alpha * resid**2 + lastVol**2 * beta)
    #     sigma = prev * tDistReturns[n]
    #     newVal = 0.01 * np.sqrt(omega + alpha * sigma**2 + prev**2 * beta)
    #     prev = newVal
    #     # arr.append(np.sqrt(prev))
    #     arr.append(prev)
    #     n = n + 1
    
    return arr


def autoregressive_model(returns):
    mod = ARIMA(returns, order = (2,0,0))
    res = mod.fit()
    constant, alpha, beta = res.params[0], res.params[1], res.params[2]
    return constant, alpha, beta

def returnsPredictive(returns, numberofData):
    #R(t+1) = constant + alpha * R(t) + epsilon
    lastVol, constant, alpha, beta, resid  = sd_estimation2(returns)
    # constant, alpha, beta = autoregressive_model(returns)
    sd_values = sd_estimation_equation2(returns, numberofData)
    zt = t_dist(returns, numberofData)
    arr = []
    n = 0
    while n < numberofData:
        if n == 0:
            prev = constant + alpha * returns.iloc[-1] + zt[n] * sd_values[n]
        newVal = constant + alpha * prev + zt[n] * sd_values[n]
        prev = newVal
        print(prev)
        arr.append(prev)
        n = n + 1
    return arr

def test(df, char, x):
    m = df[char]
    a = df[char][:-x]
    b = returnsPredictive(df[char], x)
    y = np.append(a,b)
    plt.plot(df.index,y, 'g',label = 'Future Data')
    plt.plot(df.index,m, 'b',label = 'No Future Data')
    plt.legend(loc='best')
    
    return df.index, m, y


def model(returns, typeGARCH, num, prev):
    
    model = pm.auto_arima(returns,
    
    d=0, # non-seasonal difference order
    start_p=1, # initial guess for p
    start_q=1, # initial guess for q
    max_p=5, # max value of p to test
    max_q=5, # max value of q to test                        
                        
    seasonal=False, # is the time series seasonal
                        
    information_criterion='bic', # used to select best model
    trace=True, # print results whilst training
    error_action='ignore', # ignore orders that don't work
    stepwise=True, # apply intelligent order search
                            
    )

    _arma_model = sm.tsa.SARIMAX(endog=returns,order=model.order)
    _model_result = _arma_model.fit()
    _garch_model = arch_model(_model_result.resid, mean='Zero', p=1, q=1, dist = "t", vol = typeGARCH)
    _garch_result = _garch_model.fit(disp = 'off')
    
    # https://goldinlocks.github.io/ARCH_GARCH-Volatility-Forecasting/
    index = returns.index
    start_loc = 0
    end_loc = np.where(index >= '2020-01-01')[0].min()
    forecasts = {}
    forecastsReturns = {}
    
    
    degrees_of_freedom = num - 1
    newReturn = t.rvs(degrees_of_freedom, size = num)
        
    for i in range(num):
        sys.stdout.write('-')
        sys.stdout.flush()
        res = _garch_model.fit(first_obs=start_loc + i, last_obs=i + end_loc, disp='off')
        temp = res.forecast(horizon=1, method = "simulation").variance
        print("End Loc " + str(end_loc))
        fcast = temp.iloc[i + end_loc - 1]
        forecasts[fcast.name] = fcast        
        forecastsReturns[fcast.name] = res.params.omega + res.params['alpha[1]'] * prev + np.sqrt(fcast) * newReturn[i]
        prev = res.params.omega + res.params['alpha[1]'] * prev + fcast
    print(' Done!')
    variance_fixedwin = pd.DataFrame(forecasts).T
    returns_fixedwin = pd.DataFrame(forecastsReturns).T
    
    # Calculate volatility from variance forecast with a fixed rolling window
    vol_fixedwin = np.sqrt(variance_fixedwin)
    
    # Plot results
    plt.figure(figsize=(10,5))
    
    
    # Plot volatility forecast with a fixed rolling window
    plt.plot(vol_fixedwin, color = 'red', label='Rolling Window')
    plt.plot(returns_fixedwin, color = 'blue', label='Forecasted Returns')
    plt.plot(returns.loc[variance_fixedwin.index], color = 'grey', label='Daily Return')
    
    plt.legend()
    plt.show()
    

def equityAlloc(data):

    # Example of making future returns
    # futureReturns = returnsPredictive(data,10)
    

    setReturns = data
    
    # loc, scale, shape = stats.genpareto.fit(dataRet)
    # loc, scale, shape = stats.genpareto.fit(setReturns)
    # shape, scale, loc = stats.genpareto.fit(setReturns)
    # loc, scale, shape = stats.genpareto.fit(setReturns)
    
    
    #This one works
    # loc, shape, scale = stats.genpareto.fit(setReturns)
    
    #Try this
    
    shape = 0
    loc = 0 
    scale = 0
    
    for i in range(0, 10):
        if shape == 0 and loc == 0 and scale == 0:
            shape, loc, scale = stats.genpareto.fit(setReturns)
        else:
            shape, loc, scale = stats.genpareto.fit(setReturns, shape, loc = loc, scale = scale)
    
    # shape, loc, scale = stats.genpareto.fit(setReturns, shape, loc = loc, scale = scale)
    # shape, loc, scale = stats.genpareto.fit(setReturns, shape, loc = loc, scale = scale)
    
    #stats.genpareto.fit
    #x, shape, loc, scale
    
    a = 0.95
    #What do we want to consider as a tail hedge
    #Lets just say 10%
    #beta = scale
    u = np.mean(setReturns) - 2*(np.std(setReturns)) 
    N = len(setReturns)
    Nu = len([n for n in setReturns if n >= u])
    
    
    # VaR = u + scale/shape * (((a * N/Nu) ** -shape) - 1)

    #To 
    VaR = u + scale/shape * (((a * 1/a) ** -shape) - 1)
    CVaR = (VaR + scale - shape * u) / (1 - shape)

    # VaR_old = u + scale/shape * (((a * N/Nu) ** -shape) - 1)
    # CVaR_old = (VaR_old + scale - shape * u) / (1 - shape)
    
    
    # targetCVaR = 2.07 * np.std(setReturns) - np.mean(setReturns)
    

    targetCVaR = 2.07 * np.std(setReturns) - np.mean(setReturns)
    # targetCVaR = 0.17
    maxExposure = 1
    
    
    equityAlloc = min(targetCVaR/CVaR, maxExposure)
    

    return equityAlloc