import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize as sci_minimize
from scipy.integrate import quad

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

def f_neural(theta : float, phi : np.ndarray, T : float = 1, r_max : float = 1) -> float:
  fs = fi(phi, T, r_max)
  return np.sum([f(theta) for f in fs])

def F_neural(theta : np.ndarray, phi : np.ndarray, T : float = 1, r_max : float = 1) -> np.ndarray:
  return np.array([f_neural(th, phi, T, r_max) for th in theta])

def D(phi : np.ndarray, theta : np.ndarray, F_acoustic : np.ndarray, T : float = 1, r_max : float = 1) -> float:
  return 1 - pearsonr(F_acoustic, F_neural(theta, phi, T, r_max)).statistic

def optimize_acoustic(F_acoustic : np.ndarray, phi_rand : np.ndarray, theta : np.ndarray) -> np.ndarray:
  res = sci_minimize(D, phi_rand, (theta, F_acoustic, 1, 1), method='Powell')
  if res.success:
    return res.x
  else:
    return np.array([None for phi in phi_rand])