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

#!pip install brian2hears
from brian2 import *
from brian2hears import *

from optfunc import *

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

def plotAcousticFI(azims, azimuths, freqs, itd_means, itd_means_unwp, itd_stds, fisher_info):
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
    fig.savefig('Acoustic-Fisher_Information', dpi=300)

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
  #            Download CIPIC Database             #
  ##################################################
  print('Setting HRTF DB')

  #download_cipic()

  cipic_db = load_cipic('./hrir_final.mat')

  hrir_l = np.squeeze(cipic_db['hrir_l'][:, 9, :])
  hrir_r = np.squeeze(cipic_db['hrir_r'][:, 9, :])

  ##################################################
  #              Cochlear Filter                   #
  ##################################################
  print('Applying Cochlear Filter -- Extracting ITD means')

  Sf = 44100 #sampling frequency
  trim = int(.25*Sf)

  sound = np.squeeze(2*(np.random.rand(1,4*Sf)-0.5), axis=0)

  freqs = np.arange(400, 1251, 25)#np.arange(0, 1501, 30)[1:] #np.arange(400, 1250, 100)
  azims = np.concatenate(([-80, -65, -55], np.arange(-45, 45, 5), [55, 65, 80]))

  ipd_means = np.zeros((len(azims), len(freqs)))
  ipd_stds = np.zeros_like(ipd_means)

  itd_means = np.zeros_like(ipd_means)
  itd_stds = np.zeros_like(ipd_means)

  for i, az in tqdm(enumerate(azims)):
      convolved_l = Sound(convolve(sound, hrir_l[i]), samplerate=Sf*Hz)
      convolved_r = Sound(convolve(sound, hrir_r[i]), samplerate=Sf*Hz)

      cochlear_l = Gammatone(convolved_l, freqs).process()  #TODO: ver como fazer estéreo direto (Tile/Join)
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

  #for i in range(3):
  #  itd_means_unwp[:, i] = ipd2itd(np.unwrap(ipd_means[:, i], period=np.pi*(2)), freqs[i])*1e6

  #for i in range(3, len(freqs)):
  #  itd_means_unwp[:, i] = ipd2itd(np.unwrap(ipd_means[:, i]-(2*np.pi), period=np.pi*(2)), freqs[i])*1e6


  ##################################################
  #            Acoustic Information                #
  ##################################################
  print('Calculating Acoustic Fisher Information')

  azimuths = np.arange(-80, 80, 1)

  fisher_info = np.zeros((len(azimuths), len(freqs)))
  fisher_info_ipd = np.zeros_like(fisher_info, dtype=[('IPD', np.float64), ('fi', np.float64)])

  for i in tqdm(range(len(freqs))):
    #fitar media
    mean_fit, _, _, _ = interpolate.splrep(azims, itd_means_unwp[:, i], s=len(azims), full_output=True)

    #trocar azimute -> ITD -> IPD
    #IPD = mean_fit(azimutes)*2*np.pi*freqs[i]
    ipd = interpolate.BSpline(*mean_fit)(azimuths)*2*np.pi*freqs[i]
    fisher_info_ipd['IPD'][:, i] = wrapToPi(ipd)

    #derivada da media
    dmean = interpolate.splev(azimuths, mean_fit, der=1)

    #fitar desvio padrao
    std_fit, _, _, _ = interpolate.splrep(azims, itd_stds[:, i]*1e6, s=len(azims), full_output=True)
    stds = interpolate.BSpline(*std_fit)(azimuths)
    #derivada desvio padrao
    dstd = interpolate.splev(azimuths, std_fit, der=1)

    #fisher information = (derivada da media/desvio padrao)^2 + 2*(derivada do desvio/desvio padrao)^2
    fisher_info[:, i] = np.power(dmean/stds, 2) + (2*np.power(dstd/stds, 2))

    fisher_info_ipd['fi'][:, i] = fisher_info[:, i]
    #fisher_info_ipd[:, i] = [(ipd, fi) for ipd, fi in zip(IPD, np.sqrt(fisher_info[:, i]))]

  plotAcousticFI(azims, azimuths, freqs, itd_means, itd_means_unwp, itd_stds, fisher_info)

  ##################################################
  #                 Optimization                   #
  ##################################################
  print('Beginning optimization')

  phi_optimal = []
  nneurons = 200 #200 #50

  rand =  np.random.rand(nneurons)
  phi_rand = (-np.pi * rand) + np.pi * (1 - rand)

  F_acoustic = np.sqrt(fisher_info_ipd['fi'].transpose())
  IPDs = fisher_info_ipd['IPD'].transpose()

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