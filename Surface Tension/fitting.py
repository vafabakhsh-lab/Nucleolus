from datetime import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import polarTransform
from scipy.ndimage import rotate
from scipy.special import kn
from scipy.optimize import curve_fit
from scipy.special import lpmv
from decimal import Decimal, getcontext
getcontext().prec = 30
import pyshtools as pysh
import inspect
from lmfit import Model, Parameter, report_fit
from sklearn.metrics import r2_score
import math
import random
import imageio
import tifffile as tif
from skimage import io
import statistics

from PIL import Image
from pandas import DataFrame as df
import h5py

from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.style.use(['science','nature'])
## function to generate bootstrap datasets ##
def make_bootstraps(data,n_bootstraps=100):
    #initialize output dictionary & unique value count
    dc   = {}
    unip = 0
    #get sample size
    b_size = data.shape[0]
    #get boostrap samples size
    bs_size = round(b_size)
    #get list of row indexes
    idx = [i for i in range(b_size)]
    #loop through the required number of bootstraps
    for b in range(n_bootstraps):
        #obtain boostrap samples with replacement
        sidx   = np.random.choice(idx, bs_size, replace=True)
        b_samp = data[sidx,:]
        #compute number of unique values contained in the bootstrap sample
        unip  += len(set(sidx))

        #obtain out-of-bag samples for the current b
        oidx   = list(set(idx) - set(sidx))
        o_samp = np.array([])
        if oidx:
            o_samp = data[oidx,:]
        #store results
        dc['boot_'+str(b)] = {'boot':b_samp,'test':o_samp}
    #state the mean number of unique values in the bootstraps
    #print('Mean number of unique values in each bootstrap: {:.2f}'.format(unip/n_bootstraps))
    #return the bootstrap results
    return(dc)

def fit_legendre_sigma(x, _sigma):
    global mean_R0
    global lmax
    global kappa_fit
    kb = 1.380649e-23
    T  = 310
    kappa = kappa_fit
    v_q_2 =[]
    for q in x:
        v_q = 0
        for l in range(int(q),lmax+1):
            norm_leg = pysh.legendre.legendre_lm(l, int(q), 0, 'schmidt')
            v_q += (kb*T/kappa) * ((2*l+1)/(4*np.pi))*norm_leg**2 / ((l-1)*(l+2)*(l*(l+1)+_sigma*mean_R0**2/kappa))
        v_q_2.append(v_q)
    return v_q_2

def fit_legendre(x, kappa):
    global mean_R0
    global lmax
    kb = 1.380649e-23
    T  = 310
    v_q_2 =[]
    for q in x:
        v_q = 0
        for l in range(int(q),lmax+1):
            norm_leg = pysh.legendre.legendre_lm(l, int(q), 0, 'schmidt')
            v_q += (kb*T/kappa) * ((2*l+1)/(4*np.pi))*norm_leg**2 / ((l-1)*(l+2)*(l*(l+1)))
        v_q_2.append(v_q)
    return v_q_2
def fit_legendre_two(x, _sigma, kappa):
    global mean_R0
    global lmax
    kb = 1.380649e-23
    T  = 310
    v_q_2 =[]
    for q in x:
        v_q = 0
        for l in range(int(q),lmax+1):
            norm_leg = pysh.legendre.legendre_lm(l, int(q), 0, 'schmidt')
            v_q += (kb*T/kappa) * ((2*l+1)/(4*np.pi))*norm_leg**2 / ((l-1)*(l+2)*(l*(l+1)+_sigma*mean_R0**2/kappa))
        v_q_2.append(v_q)
    return v_q_2
def Delta(lmin, lmax, kappa):
    delta = 0
    kb = 1.380649e-23
    T  = 310
    for i in range(lmin,lmax+1,1):
        delta += (0.5*kb*T/kappa) * ((2*i+1)/(i**2+i))
    return delta
def KC(l, sigma_d, delta):
    kc = 0
    kb = 1.380649e-23
    T  = 310
    for i in range(2,l,1):
        kc += (0.5*kb*T/delta) * ((2*i+1)/(i**2+i+sigma_d))
    return kc
def Delta_Exp (q_up, x, variance):
    variance =np.array([variance]).T
    x = np.array([x]).T
    print(variance.shape, x.shape)
    gn = variance[:q_up,].reshape(-1,1)
    q = x[:q_up,]
    return np.sum(gn*q**3/8)
param = np.loadtxt("./save_param.txt")
kb = 1.380649e-23
T  = 310
mean_R0 = param[0]
lmax = 242
lmin = 50
data = np.loadtxt("./save_data.txt", delimiter=',')
x=data[:,0]
variance=data[:,1]

#fitting
xx=x[lmin-2:,]
yy=variance[lmin-2:,]
xx_ =x[3:lmin-2,]
yy_ = variance[3:lmin-2,]
_popt_, _pcov_ = curve_fit(fit_legendre, xx, yy, p0 = [1e-21])
out_kappa = fit_legendre(x,_popt_)
res = (np.array(out_kappa) - np.expand_dims(variance, axis=1))/np.array(out_kappa)
plt.figure(0, figsize=(3,2))
plt.plot(x, variance, 'o', markerfacecolor='None', markeredgecolor='k')
plt.plot(x, out_kappa, 'r-', markerfacecolor='None', markeredgecolor='k')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$q$')
plt.ylabel('$\\langle|u_{q}|^2\\rangle$')
kappa_fit = np.float64(_popt_[0])
print(kappa_fit/(kb*T))
_popt_s, _pcov_s = curve_fit(fit_legendre_sigma, xx_, yy_, p0 = [50e-9])
#plt.savefig('fitting_kappa.jpg', dpi=300)
#delta_kappa = Delta(lmin, lmax, _popt_)
#excess = delta_exp + delta_kappa
#tau = (T*kb)/(excess*_popt_)
#t_plot=[]
#_tension = ((lmax**2+lmax-6*np.exp(2/tau))/(np.exp(2/tau)-1))*_popt_/mean_R0**2
out = fit_legendre_two(x, _popt_s, _popt_)
plt.plot(x, out, 'b-', markerfacecolor='None', markeredgecolor='k')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$q$')
plt.ylabel('$\\langle|u_{q}|^2\\rangle$')
plt.savefig('fitting.jpg', dpi=300)

print(_popt_s, _popt_/(kb*T))
save_data = np.concatenate([_popt_s, _popt_/(kb*T)])
np.savetxt('result_fitting.txt', save_data, delimiter=",")
#checking R^2 residual
r_squared = r2_score(out, variance)
print(r_squared)
plt.show()
