"""
File: authors_code/ECEI_MT.py

Purpose:
Supplied reference implementation of the ECEI (feasibility-infeasibility weighted
Expected Improvement) algorithm on the Modified Townsend benchmark.

Role in the experiment:
External baseline implementation from the original authors' Code.zip, used
as an exact numerical reference to verify the project's ECEI acquisition and fallbacks.

Key details:
- Implements ECEI_acq with main branch (flag=1) and fallback branch (flag=2).
- Used read-only by test_fidelity.py and tests/test_suite.py via dynamic extraction.
- Uses atan2(x1, x2 + 1e-9) in constraint definition.
"""
# load required packages
# Constrained Bayesian Optimization with Feasibility‑Infeasibility Weighted Improvement Criterion
import numpy as np
import warnings
import argparse

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from scipy.stats import qmc, norm
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
np.random.seed(27)

# set dimension, and bounds
global dim, bounds, unit_bounds
dim = 2
l_bounds = [-2.25, -2.5]
u_bounds = [2.5, 1.75]
bounds = np.dstack((l_bounds,u_bounds))[0]
unit_bounds = np.dstack(([0]*dim,[1]*dim))[0]

# objective function
def objective(vec, noise=0.1):
    return -(np.cos((vec[0]-0.1)*vec[1]))**2- vec[0]*(np.sin(3*vec[0]+vec[1]))

# Constraint function
def constraint(vec):
    t=np.arctan2(vec[0],(vec[1]+1E-9))
    return (vec[0]**2) + (vec[1]**2)-(2*np.cos(t)-0.5*np.cos(2*t)-0.25*np.cos(3*t)-0.125*np.cos(4*t))**2 -(2*np.sin(t))**2

# surrogate prediction
def surrogate(model, X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.predict(X, return_std = True)
    
def normalize(X, bounds):
    if X.ndim == 1:
        X = X.reshape(1, -1)  # Convert 1D array to 2D matrix with 1 row
    Xnorm = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    return Xnorm

def unnormalize(X, bounds):
    if X.ndim == 1:
        X = X.reshape(1, -1)  # Convert 1D array to 2D matrix with 1 row
    Xunnorm = X * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    return Xunnorm

# Acquistion function
def ECEI_acq(Xsamples, best, M, fmean, fsd, modelf, modelc, flag):
    # compute the acquisition function
    muf, stdf = surrogate(modelf, Xsamples)
    muc, stdc = surrogate(modelc, Xsamples)

    muf = muf*fsd + fmean
    stdf = stdf*fsd
    muf1 = muf + M

    theta = 1-norm.cdf(-muc/(stdc+1e-9))
    
    if flag == 1:
        Z1 = (best-muf1)/(stdf+1e-9)
        Z2 = (best-muf)/(stdf+1e-9)
        probs = ((best-muf1)*norm.cdf(Z1)+stdf*norm.pdf(Z1))*theta + ((best-muf)*norm.cdf(Z2)+stdf*norm.pdf(Z2))*(1-theta)
        return -probs
    else:
        return muf + M*theta 

# optimize the acquistion function
def opt_acq(best,M,fmean,fsd,modelf,modelc,l):
    flag = 1

    if l%5==0:
        num = 10000
    else:
        num = 1000
    
    def acq_fun(Xsamples):
        return ECEI_acq(Xsamples.reshape(-1,dim), best, M, fmean, fsd, modelf, modelc, flag)
    
    sampler = qmc.LatinHypercube(d=dim,seed=l)
    Xsamples = sampler.random(n=num)
    scores = acq_fun(Xsamples)

    sum_scores = sum(-scores> 0)#1e-16)

    if sum_scores <= num*(1e-2):
        flag = 2
        scores = acq_fun(Xsamples)
        ix = np.argmin(scores)
        min_acq = scores.min()
        best_x = Xsamples[ix]
        res = minimize(fun=acq_fun,
                  x0=best_x,
                  bounds=unit_bounds,
                  method="L-BFGS-B")
        if res.fun < min_acq:
            best_x = res.x
            min_acq = res.fun
    else:
        flag = 1
        min_acq = scores.min()
        ix = np.argmin(scores)
        best_x = Xsamples[ix]
        res = minimize(fun=acq_fun,
                  x0=best_x,
                  bounds=unit_bounds,
                  method="L-BFGS-B")
        if res.fun < min_acq:
            best_x = res.x
            min_acq = res.fun
    return best_x

# find the best and parameters
def best_params(f,c):

    # M = max(f) - min(f)
    M = max(abs(f))
    indict = np.where(c<=0, 1, 0)

    best = min(f+M*(1-np.array(indict)))

    return best, M

# main BO loop
def BO_Ind(X,f,c,K):
    kernel0 = Matern(length_scale=np.ones(dim),length_scale_bounds=[(1/2,8.0)]*dim,nu=2.5)
    kernel1 = Matern(length_scale=np.ones(dim),length_scale_bounds=[(1/2,8.0)]*dim,nu=2.5)

    modelf = GaussianProcessRegressor(kernel0)
    modelc = GaussianProcessRegressor(kernel1)

    fmean= np.mean(f)
    fsd = np.std(f)
    ff = (f-fmean)/fsd
    cc = np.sign(c)*(np.log(abs(c)+1))

    k = 0

    while(k<K):

        norm_X = normalize(X,bounds)

        modelf.fit(norm_X,ff)
        modelc.fit(norm_X,cc)

        best, M = best_params(f,cc)

        x = opt_acq(best,M,fmean,fsd,modelf,modelc,k+1)

        unorm_x = unnormalize(x, bounds)[0]
        
        f1 = objective(unorm_x)
        c1 = constraint(unorm_x)

        X = np.vstack((X,unorm_x))
        f = np.hstack((f,f1))
        c = np.hstack((c,c1))

        fmean = np.mean(f)
        fsd = np.std(f)
        ff = (f-fmean)/fsd
        cc = np.hstack((cc, np.sign(c1)*(np.log(abs(c1)+1))))
        
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
    print(f"ECEIBO: iteration= {i+1}, optimal = {f1[idx][id]}")

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
np.savetxt('prog_MT_ECEI.csv',fx,delimiter=',',fmt='%f')
