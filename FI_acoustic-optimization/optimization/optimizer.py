import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from tqdm import tqdm
import pickle
import urllib
from multiprocessing import Pool, cpu_count
from itertools import repeat as irepeat
from time import time as pytime

from scipy.optimize import minimize as sci_minimize
from scipy.optimize import basinhopping
from scipy.integrate import quad
from scipy import interpolate
from scipy.io import loadmat
from scipy.stats import circmean, circstd, spearmanr
from scipy.signal import hilbert, convolve

from optfunc import *

def plotHist(phi_optimal, nneurons):
  t = np.linspace(-np.pi-0.1, np.pi+0.1, 50)

  x_ = []
  for phi in phi_optimal:
    phi_wrap = (phi+np.pi) % (2*np.pi) - np.pi #wrap to pi
    x, _, _ = plt.hist(phi_wrap, bins=t)
    x_.append(x)

  x_ = np.asarray(x_)

  fig, ax = plt.subplots(figsize=(8,8))
  im = ax.imshow(np.flip(x_), cmap='turbo',
               aspect='equal', interpolation = 'none', interpolation_stage='data')
  #ax.set_xticks([0.,  24.,  48.], labels=[-3.14, 0, 3.14], minor=False)
  #ax3.set_yticks([0., 10., 20., 30., 40., 49.], labels=[1500, 1200, 900, 600, 300, 30])
  ax.set_xlabel('Best IPD (phase)')
  ax.set_ylabel('Frequency (Hz)')
  ax.set_title('Human head')
  fig.suptitle('Neuronal best phase for each sound frequency matched with Acoustic FI', fontsize='x-large')
  fig.colorbar(im, label='# of neurons out of {}'.format(nneurons))
  fig.savefig('Neural-Acoustic-Matched_Best_IPD', dpi=300)

if __name__ == '__main__':
    ##################################################
    #                 Optimization                   #
    ##################################################
    print('Beginning optimization')

    F_acoustic = None
    with open('../FI_all-subjects/fi_allsubjects.pkl', 'rb') as f:
       F_acoustic = pickle.load(f)
    
    IPDs = None
    with open('../FI_all-subjects/IPD_allsubjects.pkl', 'rb') as f:
       IPDs = pickle.load(f)

    limits = None
    with open('../FI_all-subjects/IPDlimits_allsubjects.pkl', 'rb') as f:
       limits = pickle.load(f)

    phi_optimal = []
    nneurons = 200 #50

    rand =  np.random.rand(nneurons)
    phi_rand = (-np.pi * rand) + np.pi * (1 - rand)

    #F_acoustic = np.sqrt(fisher_info_ipd['fi'].transpose())
    #IPDs = fisher_info_ipd['IPD'].transpose()
    limits = np.array(limits)

    F_acoustic = np.sqrt(np.array(F_acoustic))
    F_acoustic = [fa[np.where(lim == True)] for fa, lim in zip(F_acoustic, limits)]
    IPDs = [ipd[np.where(lim == True)] for ipd, lim in zip(IPDs, limits)]

    phi_optimal = None

    start_time = pytime()
    with Pool(cpu_count()) as p:
        phi_optimal = p.starmap(optimize_acoustic, zip(F_acoustic, irepeat(phi_rand), IPDs))
    #phi_optimal.append(res)
    stop_time = pytime()

    phi_opt = open('Acoustic_Optimization', 'ab')
    pickle.dump(phi_optimal, phi_opt)
    phi_opt.close()

    plotHist(phi_optimal, nneurons)

    print(f'Optimization concluded - Elapsed time : {stop_time - start_time}')