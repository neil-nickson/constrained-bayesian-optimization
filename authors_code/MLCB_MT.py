"""
File: authors_code/MLCB_MT.py

Purpose:
Supplied reference implementation of the CBO-LCB (Maximum-probability Lower
Confidence Bound) algorithm on the Modified Townsend benchmark.

Role in the experiment:
External baseline implementation from the original authors' Code.zip, used
as an exact numerical reference to verify the project's CBO-LCB acquisition.

Key details:
- Implements MLCB_acq, objective, and constraint functions.
- Used read-only by test_fidelity.py and tests/test_suite.py via dynamic extraction.
"""
# load required packages
globals().clear()
import numpy as np
import warnings
import time
import argparse

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

from scipy.stats import qmc, norm
from scipy.optimize import minimize


warnings.filterwarnings("ignore")
np.random.seed(27)

# set dimension, and bounds
global dim, l_bounds, u_bounds
dim = 2
l_bounds = [-2.25, -2.5]
u_bounds = [2.5, 1.75]

# objective function
def objective(vec, noise=0.1):
    return -(np.cos((vec[0]-0.1)*vec[1]))**2- vec[0]*(np.sin(3*vec[0]+vec[1]))

# Constraint function
def constraint(vec):
    t=np.arctan2(vec[0],(vec[1]))
    return (vec[0]**2) + (vec[1]**2) - (2*np.cos(t)-0.5*np.cos(2*t)-0.25*np.cos(3*t)-0.125*np.cos(4*t))**2 -(2*np.sin(t))**2
 
# surrogate prediction
def surrogate(model, X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.predict(X, return_std = True)

# Acquistion function
def MLCB_acq(Xsamples, M, modelf, modelc):
    muf, stdf = surrogate(modelf, Xsamples)
    muc, stdc = surrogate(modelc, Xsamples)

    lambda1 = 1/norm.cdf(0.05)
    pc = norm.cdf(-muc/(stdc+1e-9))
    
    muy = muf + M*(1-pc)
    stdy = np.sqrt(stdf**2+(M**2)*pc*(1-pc))
    probs = muy - lambda1*stdy

    return probs

# optimize the acquistion function
def opt_acq(M,modelf,modelc,l):
    if l%5==0:
        num = 10000
    else:
        num = 1000
    
    def acq_fun(Xsamples):
        return MLCB_acq(Xsamples.reshape(-1,dim), M, modelf, modelc)
    
    bounds = np.dstack((l_bounds, u_bounds))[0]

    sampler = qmc.LatinHypercube(d=dim,seed=l)
    sampleX = sampler.random(n = num)
    Xsamples = qmc.scale(sampleX, l_bounds, u_bounds)

    scores = acq_fun(Xsamples)
    ix = np.argmin(scores)

    best_x = Xsamples[ix]
    min_acq = scores.min()
    res = minimize(fun=acq_fun,
                  x0=best_x,
                  bounds=bounds,    
                  method="L-BFGS-B")
    if res.fun < min_acq:
        best_x = res.x
        min_acq = res.fun
    
    return best_x

# main BO loop
def BO_Ind(X,f,c,K):
    kernel0 = Matern(length_scale=np.ones(dim),nu=2.5)
    kernel1 = Matern(length_scale=np.ones(dim),nu=2.5)

    modelf = GaussianProcessRegressor(kernel0)
    modelc = GaussianProcessRegressor(kernel1)
    
    k = 0

    while(k<K):
        # print(k)

        modelf.fit(X,f)
        modelc.fit(X,c)

        M = max(abs(f))

        x = opt_acq(M,modelf,modelc,k+1)
        
        f1 = objective(x)
        c1 = constraint(x)

        X = np.vstack((X,x))
        f = np.hstack((f,f1))
        c = np.hstack((c,c1))
        
        k = k+1
    return X,f,c

# simulation
# K    = 100
# simu = 3
# n    = 20

parser = argparse.ArgumentParser()
parser.add_argument('arg1', type=int)
parser.add_argument('arg2', type=int)
parser.add_argument('arg3', type=int)
args = parser.parse_args()

n = args.arg1    # Number of Initial sample 
K = args.arg2    # Number of addtional sample
simu = args.arg3 # Number of simulation

data = np.loadtxt("MT_Xsamples.csv", delimiter=",")

XX   = []
ff   = []
cc   = []


for i in range(simu):
    start = i * n
    end = start + n
    X = data[start:end]

    f = np.asarray([objective(z) for z in X])
    c = np.asarray([constraint(z) for z in X])
    x1,f1,c1 = BO_Ind(X,f,c,K)

    idx = np.where(c1<=0)
    id = np.argmin(f1[idx])
    print(f"M-LCBBO: iteration= {i+1}, optimal = {f1[idx][id]}")

    XX.append(x1)
    ff.append(f1)
    cc.append(c1)

# 
fx = np.copy(ff)
for i in range(np.shape(fx)[0]):
    idxx = np.where(np.asarray(cc[i])<=0)
    if idxx[0][0]!=0:
        fx[i][0:idxx[0][0]] = np.inf
    hx = np.zeros(np.shape(fx)[1])
    hx[idxx] = ff[i][idxx[0]]
    for j in range(np.shape(fx)[1]-1):
        if ff[i][j+1] < fx[i][j] and hx[j+1]!=0:
            fx[i][j+1] = ff[i][j+1]
        else:
            fx[i][j+1] = fx[i][j]

# save result for further analysis
np.savetxt('prog_MT_MLCB.csv',fx,delimiter=',',fmt='%f')

