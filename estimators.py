#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 21:08:29 2021

@author: apple
"""

import numpy as np


#Estimator for variance of returns
def sigma_est(Sigma_scm,mu_sm):
    Sigma_T = np.diag(np.diag(Sigma_scm))
    T = len(Sigma_scm)
    W = np.eye(T) - np.ones(T*T).reshape((T, T))/T
    rho1_sweep = np.exp(np.linspace(-10, 10, 100))
    obj = []
    for rho1 in rho1_sweep:
        Sigma_sh = rho1 * Sigma_T + Sigma_scm
        D = (1/T)*np.sum(np.diag(Sigma_scm @ np.linalg.solve(Sigma_sh, np.eye(len(Sigma_sh)))))
        delta = D / (1 - D)
        B = np.linalg.solve((np.eye(T)+delta*W) @ (np.eye(T)+delta*W), np.eye(len(W)))
        b = T / np.sum(np.diag(W @ B))
        inv_S_sh_mu = np.linalg.solve(Sigma_sh, mu_sm).reshape((1, Sigma_sh.shape[0]))
        num = np.sum(mu_sm * inv_S_sh_mu) - delta
        den = np.sqrt(b * inv_S_sh_mu @ Sigma_scm @ inv_S_sh_mu.T)
        obj.append(num/den)
    i_max = np.argmax(obj)
    rho1 = rho1_sweep[i_max]
    rho_1 = (rho1/(1 + rho1))
    estSigma = (1-rho_1)*Sigma_scm + rho_1*Sigma_T
    return estSigma, rho_1

#Estimators for mean of returns
def mu_est(sigma,mu):
    T = len(sigma)
    t_1 = np.zeros(T)
    lambdas , _ = np.linalg.eig(sigma)
    lmd_max = np.max(lambdas)
    lmd_mean = np.mean(lambdas)
    rho_2 = (1/T)*(T*lmd_mean - 2*lmd_max)/np.linalg.norm(mu - t_1)**2
    estMu = rho_2 * mu + (1-rho_2) * np.sum(mu) * np.ones(T) / T
    return estMu,rho_2

