import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
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
from scipy.stats import circmean, circstd, spearmanr, pearsonr
from scipy.signal import hilbert, convolve

from brian2 import *
from brian2hears import *

def download_cipic(subject='058'):
  url = f"https://raw.githubusercontent.com/amini-allight/cipic-hrtf-database/master/standard_hrir_database/subject_{subject}/hrir_final.mat"
  f = urllib.request.urlopen(url)
  with open('hrir_final.mat', 'wb') as data:
    data.write(f.read())

def load_cipic(filename='./hrir_final.mat'):
  return loadmat(filename)

def download_matsound():
  url = "https://raw.githubusercontent.com/mamebios/neural_optimization/master/demo_FIacustico/y.mat"
  f = urllib.request.urlopen(url)
  with open('y.mat', 'wb') as data:
    data.write(f.read())

def load_soundmat(filename='y.mat'):
  return loadmat(filename)

def wrapToPi(x):
  return (x+np.pi)%(2*np.pi)-np.pi

def ipd2itd(ipd, freq):
  return ipd/(freq*2*np.pi)

def itd2ipd(itd, freq):
  return itd*(2*np.pi*freq)

def plotAcousticFI(azims, azimuths, freqs, itd_means, itd_means_unwp, itd_stds, fisher_info, savepath=''):
    fig, axd = plt.subplot_mosaic([['upper left', 'center', 'right'],
                               ['lower left', 'center', 'right']],
                              figsize=(14, 7), layout="constrained")
    for k, ax in axd.items():
        if k == 'upper left':
            for i in range(len(freqs)):
                ax.plot(azims, itd_means[:, i]*1e6, '-', label=freqs[i])
            ax.set_title('ITD means')
            ax.set_ylabel(r'Time ($\mu$s)')
            ax.set_xlabel('Azimuth')
            #ax.legend()
            ax.grid()

        elif k == 'lower left':
            for i in range(len(freqs)):
                ax.plot(azims, itd_means_unwp[:, i], '-', label=freqs[i])
            ax.set_title('unwraped ITD means')
            #ax.ylabel(r'Time ($\mu$s)')
            ax.set_xlabel('Azimuth')
            #ax.legend()
            ax.grid()

        elif k == 'center':
            for i in range(len(freqs)):
                ax.plot(azims, itd_stds[:, i]*1e6, '-')
            ax.set_title('ITD standard deviations')
            ax.set_xlabel('Azimuth')
            ax.grid()

        elif k == 'right':
            for i in range(len(freqs)):
                ax.plot(azimuths, fisher_info[:, i], '-', label=freqs[i])
            ax.set_title('Acoustic Fisher Information')
            ax.set_xlabel('Azimuth')
            ax.grid()

    fig.suptitle('Sound stimulus Azimuthal Fisher Information')
    fig.savefig(savepath+'acoustic_FI-graphic', dpi=300)

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

if __name__ == '__main__':
    ITD_means = []
    ITD_means_unwp = []
    FI_ = []
    FI_ipd = []

    path = '../cipic-hrtf-database/standard_hrir_database/'

    subjects = [s for s in os.listdir(path) if s[:7] == 'subject']

    for subject in subjects:
        print('='*50)
        print(subject)
        ##################################################
        #            Setup CIPIC Database                #
        ##################################################
        print('Setting HRTF DB')

        cipic_db = load_cipic(path+'{}/hrir_final.mat'.format(subject))

        hrir_l = np.squeeze(cipic_db['hrir_l'][:, 9, :])
        hrir_r = np.squeeze(cipic_db['hrir_r'][:, 9, :])

        ##################################################
        #              Cochlear Filter                   #
        ##################################################
        print('Applying Cochlear Filter -- Extracting ITD means')

        Sf = 44100 #sampling frequency
        trim = int(.25*Sf)

        sound = np.squeeze(2*(np.random.rand(1,4*Sf)-0.5), axis=0)

        freqs = np.arange(250, 1251, 25)#np.arange(400, 1250, 25)#np.arange(0, 1501, 30)[1:] #np.arange(400, 1250, 100)
        azims = np.concatenate(([-80, -65, -55], np.arange(-45, 45, 5), [55, 65, 80]))

        ipd_means = np.zeros((len(azims), len(freqs)))
        ipd_stds = np.zeros_like(ipd_means)

        itd_means = np.zeros_like(ipd_means)
        itd_stds = np.zeros_like(ipd_means)

        for i, az in tqdm(enumerate(azims)):
            convolved_l = Sound(convolve(sound, hrir_l[i]), samplerate=Sf*Hz)
            convolved_r = Sound(convolve(sound, hrir_r[i]), samplerate=Sf*Hz)

            cochlear_l = Gammatone(convolved_l, freqs).process()  #TODO: ver como fazer estereo direto (Tile/Join)
            cochlear_r = Gammatone(convolved_r, freqs).process()

            for cl, cr, j in zip(cochlear_l.T, cochlear_r.T, range(len(freqs))):  # iterate over cochelar filter center freqs
                transformed = hilbert(np.stack((cl, cr)), axis=1)

                ipd = wrapToPi(np.diff(np.angle(transformed), axis=0))[0, :]
                ipd = ipd[trim:-trim]

                ipd_mean = circmean(ipd, high=np.pi, low=-np.pi)
                ipd_std = circstd(ipd, high=np.pi, low=-np.pi)

                ipd_means[i,j] = ipd_mean
                ipd_stds[i,j] = ipd_std

                itd_means[i,j] = ipd2itd(ipd_mean, freqs[j])
                itd_stds[i,j] = ipd2itd(ipd_std, freqs[j])

        itd_means_unwp = np.zeros((len(azims), len(freqs)))

        for i in range(len(freqs)):
            if(np.allclose(np.unwrap(ipd_means[:, i], period=np.pi*2), ipd_means[:, i])):
                itd_means_unwp[:, i] = ipd2itd(np.unwrap(ipd_means[:, i], period=np.pi*(2)), freqs[i])*1e6
            else:
                itd_means_unwp[:, i] = ipd2itd(np.unwrap(ipd_means[:, i]-(2*np.pi), period=np.pi*(2)), freqs[i])*1e6

        ITD_means.append(itd_means)
        ITD_means_unwp.append(itd_means_unwp)

        ##################################################
        #     Acoustic Information for all subjects      #
        ##################################################
        print('Calculating Acoustic Fisher Information')

        azimuths = np.arange(-80, 80, 1)

        fisher_info = np.zeros((len(azimuths), len(freqs)))
        fisher_info_ipd = np.zeros_like(fisher_info, dtype=[('IPD', np.float64), ('fi', np.float64)])

        for i in tqdm(range(len(freqs))):
            #fitar media
            mean_fit = interpolate.CubicSpline(azims, itd_means_unwp[:, i])

            #trocar azimute -> ITD -> IPD
            ipd = mean_fit(azimuths)*2*np.pi*freqs[i]*1e-6
            fisher_info_ipd['IPD'][:, i] = ipd

            #derivada da media
            dmean = mean_fit.derivative(nu=1)(azimuths)

            #fitar desvio padrao
            std_fit = interpolate.CubicSpline(azims, itd_stds[:, i]*1e6)
            stds = std_fit(azimuths)

            #derivada desvio padrao
            dstd = std_fit.derivative(nu=1)(azimuths)

            #fisher information = (derivada da media/desvio padrao)^2 + 2*(derivada do desvio/desvio padrao)^2
            fisher_info[:, i] = np.power(dmean/stds, 2) + (2*np.power(dstd/stds, 2))

            fisher_info_ipd['fi'][:, i] = fisher_info[:, i]

        FI_.append(fisher_info)
        FI_ipd.append(fisher_info_ipd)

        plotAcousticFI(azims, azimuths, freqs, itd_means, itd_means_unwp, itd_stds, fisher_info, './subjects/subject_{}_'.format(subject))


    ##################################################
    #          Median Acoustic Information           #
    ##################################################

    FIarrays = np.array([sbj['fi'] for sbj in FI_ipd])
    IPDarrays = np.array([sbj['IPD'] for sbj in FI_ipd])

    FImedians = []
    IPDmedians = []
    IPDlimits = []

    azimuths = np.arange(-80, 80, 1)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,8))

    for i in range(len(freqs)):

        FImedian = np.median(FIarrays[:, :, i], axis=0)
        FImedians.append(FImedian)

        IPDmedian = np.median(IPDarrays[:, :, i], axis=0)
        IPDmedians.append(IPDmedian)

        mask = np.logical_and(np.greater_equal(IPDmedian, -np.pi), np.less_equal(IPDmedian, np.pi))
        IPDlimits.append(mask)

        ax1.plot(azimuths[np.where(mask == True)], FImedian[np.where(mask == True)])
        ax2.plot(azimuths[np.where(mask == True)], IPDmedian[np.where(mask == True)])

    fig.suptitle('Median Acoustic Fisher Information from all subjects')
    fig.savefig('ALL_subjects_median_acoustic_FI-graphic', dpi=300)

    with open('fi_allsubjects.pkl', 'wb') as f:
        pickle.dump(FImedians, f)

    with open('IPD_allsubjects.pkl', 'wb') as f:
        pickle.dump(IPDmedians, f)

    with open('IPDlimits_allsubjects.pkl', 'wb') as f:
        pickle.dump(IPDlimits, f)