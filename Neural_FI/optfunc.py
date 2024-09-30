import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import minimize as sci_minimize
from scipy.integrate import quad

# Tunning Curve for the theta IPD conditioned to a neuron's phi paramether
def g(theta : float, phi_i : float, r_max : float = 1) -> float:
  return r_max * (np.cos((theta - phi_i)/2) + 0.5)**4

# Single neuron's Fisher Information function
# It's phi value is parameterized
def fi(phi : np.ndarray, T : float = 1, r_max : float = 1) -> list:
  fs = []
  for phi_i in phi:
    intermediate = lambda p_i : lambda theta : T*r_max*((np.cos(theta - p_i) + 1)**2)*np.sin(theta - p_i)**2
    fs.append(intermediate(phi_i))

  return fs

# Function for the inverse (1 over) the population's Fisher Information function
# The population's FI function is obtained by the sum of the FI functions of all neurons over every phi value
def F(theta : float, phi : np.ndarray, T : float = 1, r_max : float = 1) -> float:
  fs = fi(phi, T, r_max)
  return 1/np.sum([f(theta) for f in fs])

# Cost function over the population phi parameters 
# obtained by the integral of 1/F, where F is the population's Fisher Information
# the integration limits delimits the values for theta
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
  return 1 - spearmanr(F_acoustic, F_neural(theta, phi, T, r_max)).statistic

def optimize_acoustic(F_acoustic : np.ndarray, phi_rand : np.ndarray, theta : np.ndarray) -> np.ndarray:
  res = sci_minimize(D, phi_rand, (theta, F_acoustic, 1, 1), method='Powell')
  if res.success:
    return res.x
  else:
    return np.array([None for phi in phi_rand])