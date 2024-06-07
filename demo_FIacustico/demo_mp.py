import multiprocessing as mp
import time

'''import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sb
import sympy as s
from tqdm import tqdm
import pickle
from scipy.optimize import minimize as sci_minimize
from scipy.optimize import basinhopping
from scipy.integrate import quad
from sympy import pycode
s.init_printing()''';

def square(x):
    return x*x


if __name__ == '__main__':

    mylist = [i for i in range(10000)]#[1,2,3,4,5]#

    ti = time.time()
    with mp.Pool(5) as p:
        print(p.map(square, mylist))

    tf = time.time()
    print(tf - ti)