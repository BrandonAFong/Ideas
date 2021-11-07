#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 21:15:58 2021

@author: apple
"""

import numpy as np
import pandas_datareader as pdr

def dataReturn(stocks,startDate,endDate, logReturnFlags):
    
    df = pdr.get_data_yahoo(stocks,start = startDate, end = endDate)
    
    if logReturnFlags:
        df_new = df['Adj Close'].apply(np.log).apply(np.diff)
        df_new.index = df.index[1:]
        return df_new
    else:
        return df['Adj Close']
    
#Converts dataframe to a matrix
def df_to_matrix(df):
    
    mat = df.values
    
    return mat

#Indexing Correlation Matrix
def indexCorrMatrix(df):
    
    dfCorr = df.corr(method = 'pearson')
    columnDic = {k:v for v, k in enumerate(dfCorr.columns)}
    
    df_new = dfCorr.rename(columns = columnDic, index = columnDic)
    
    return df_new, columnDic


def indexCovMatrix(df):
    dfCov = df.cov()
    columnDic = {k:v for v, k in enumerate(dfCov.columns)}
    df_new = dfCov.rename(columns = columnDic, index = columnDic)
    return df_new, columnDic

def cumulativeRet(weights, ret, n=1):
    
    weightedReturns = (weights * ret)
    portRet = weightedReturns.sum(axis=1)
    cumulativeRet = (n * portRet + 1).cumprod()

    return cumulativeRet

def portRet(weights, ret, n = 1):
    weightedReturns = (n * weights * ret)
    portRet = weightedReturns.sum(axis=1)

    return portRet

def drawdown(r):
    rollMax = r.cummax()
    daily_drawdown = r/rollMax -1
    Max_Daily_Drawdown = daily_drawdown.cummin()
    return Max_Daily_Drawdown

def sharpe_ratio(return_series, N = 255, rf = 0.01):
    mean = return_series.mean() * N -rf
    sigma = return_series.std() * np.sqrt(N)
    return mean / sigma

def sortino_ratio(series, N=255,rf = 0.01):
    mean = series.mean() * N -rf
    std_neg = series[series<0].std()*np.sqrt(N)
    return mean/std_neg
