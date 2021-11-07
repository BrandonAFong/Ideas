#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 22:46:54 2021

@author: apple
"""

import numpy as np 
import pandas as pd


from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from backtest import df_to_matrix,indexCovMatrix

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

