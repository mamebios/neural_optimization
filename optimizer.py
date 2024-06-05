import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sb
import sympy as s
from tqdm import tqdm
import pickle
from scipy.optimize import minimize as sci_minimize
from scipy.optimize import basinhopping
from scipy.integrate import quad
from sympy import pycode
import multiprocessing as mp
from itertools import repeat
import time

# Tunning Curve para o IPD de theta condicionada a phi do neuronio
def g(theta : float, phi_i : float, r_max : float = 1) -> float:
    return r_max * (np.cos((theta - phi_i)/2) + 0.5)**4

# Funcao da informacao de Fisher de um neuronio
# Seu valor de phi eh parametrizado
def fi(phi : np.ndarray, T : float = 1, r_max : float = 1) -> list:
    fs = []
    for phi_i in phi:
        intermediate = lambda p_i : lambda theta : T*r_max*((np.cos(theta - p_i) + 1)**2)*np.sin(theta - p_i)**2
        fs.append(intermediate(phi_i))

    return fs

# Funcao de 1 sobre a Funcao da informacao de Fisher da populacao
# Eh obtida pela soma das funcoes de informacao de fisher para cada valor de phi
def F(theta : float, phi : np.ndarray, T : float = 1, r_max : float = 1) -> float:
    fs = fi(phi, T, r_max)
    return 1/np.sum([f(theta) for f in fs])

# Funcao custo em funcao dos parametros phi da populacao
# obtida pela integracao de 1/F, em que F eh a informacao de Fisher da populacao
# os limites de integracao delimitam o theta
def V(phi : np.ndarray, lim : float, T : float = 1, r_max : float = 1) -> float:
    return quad(F, -lim, lim, args=(phi, T, r_max))[0]

def optimize(lim, phi_rand):
    res = sci_minimize(V, phi_rand, (lim, 1, 1), method='Powell')
    if res.success:
        return res.x
    else:
        return [None for phi in phi_rand]

#############
class Experimento():
    def __init__(self, N = 200, tamanho_cabeca = 15e-2, Nfreqs = 50):
        self.N = N

        self.frequencias = np.arange(0, 1501, Nfreqs)[1:]
        self.limites = self.frequencias * 2 * np.pi * (tamanho_cabeca/320.29)

        self.rand =  np.random.rand(self.N)
        self.phi_rand = (-np.pi * self.rand) + np.pi * (1 - self.rand)

        self.thetas = np.arange(-np.pi, np.pi, 0.01)/(2*np.pi)

        self.phi_optimal = None

    def run(self):
        self.phi_optimal = []
        for lim in tqdm(self.limites):
            res = sci_minimize(V, self.phi_rand, (lim, 1, 1), method='Powell')
            if res.success:
                self.phi_optimal.append(res.x)
            else:
                self.phi_optimal.append(self.phi_rand*0)

    def plot(self, lim, t = [], fig = None, ax = None):
        if self.phi_optimal:
            if not np.any(t):
                t = np.copy(self.thetas)
                t = t*2*np.pi

            if not ax:
                fig, ax = plt.subplots(layout="constrained")

            for phi in self.phi_optimal[lim]:
                ax.plot(t, g(t, phi, 1))
                #plt.plot(thetas, [g(thetas, phi, 1) for phi in phis]) #g(thetas, phi, 1)
            ax.vlines(self.limites[lim], 0, 5, 'k')
            ax.vlines(-self.limites[lim], 0, 5, 'k')
            ax.set_title("limite: {:.2f}".format(self.limites[lim]))

    def plotAll(self):
        nGrafs = int(np.ceil(exp.limites.shape[0]/3))
        fig, axs = plt.subplots(nGrafs, 3, layout="constrained")

        t = np.copy(self.thetas)
        t = t*2*np.pi
        for lim, ax in zip(range(self.limites.shape[0]), axs.flat):
            self.plot(lim, t, fig, ax)

        for ax in axs.flat:
            if not ax.has_data():
                ax.set_axis_off()


    def load(self, path):
        file = open(path, 'rb')
        self.phi_optimal = pickle.load(file)
        file.close()
            
    def save(self, name='phi_opt'):
        phi_opt = open(name, 'ab')
        pickle.dump(self.phi_optimal, phi_opt)
        phi_opt.close()

if __name__ == '__main__':

    exp = Experimento(N=200, tamanho_cabeca=15e-2, Nfreqs=30)

    ti = time.time()
    with mp.Pool(8) as p:
        exp.phi_optimal = p.starmap(optimize, zip(exp.limites, repeat(exp.phi_rand)))

    exp.save('optimization-FI_neural-human_head')

    tf = time.time()
    print(tf - ti)