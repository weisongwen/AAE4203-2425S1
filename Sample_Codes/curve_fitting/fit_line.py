import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

def func(p, x):
    k,b = p
    return k * x + b

def error(p, x, y):
    return func(p, x) - y

def fit_line(points):
    x = points[:, 0]
    y = points[:, 1]
    p0=[1,20]
    Para=leastsq(error,p0,args=(x,y))
    k, b = Para[0]
    print("line function: y =",k," * x + ",b)
    return k, b