import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

def func(p, x):
    k,b = p
    return k * x + b

def error_vertical(p, x, y):
    return func(p, x) - y

def error_perpendicular(p, x, y):
    a = p[0]
    b = p[1]
    # perpendicular offsets
    return (a * x + b - y) / np.sqrt(a**2 + 1) 

def fit_line_LLSE(points):
    x = points[:, 0]
    y = points[:, 1]
    initial_guess = [1,1]
    Para=leastsq(error_vertical,initial_guess,args=(x,y))
    k, b = Para[0]
    print("line function: y =",k," * x + ",b)
    return k, b

def fit_line_LSE(points):
    x = points[:, 0]
    y = points[:, 1]
    initial_guess = [1,1]
    Para=leastsq(error_perpendicular,initial_guess,args=(x,y))
    k, b = Para[0]
    print("line function: y =",k," * x + ",b)
    return k, b