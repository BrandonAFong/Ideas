#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 22:48:21 2021

@author: apple
"""

import numpy as np 
import pandas as pd
from HRP import seriation

import fastcluster
from scipy.cluster.hierarchy import fcluster

from gap_statistic import OptimalK

from backtest import df_to_matrix

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
    