import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.optimize import curve_fit

# Specifies the shape of the function. In this case, the objective equation is a quadratic equation.
def curve_func(p, x):              
    a,b,c = p                      
    return a * x**2 + b * x + c  

# The error curve is the difference between the actual value and the predicted value.
def error_curve(p, x, y):
    dist_err = curve_func(p, x) - y
    return dist_err

# Fit the curve using the least squares method
def fit_curve(points):
    # Sample data points (Xi,Yi), need to be converted to array (list) form.
    x = points[:, 0]
    y = points[:, 1]
    # The initial value of the coefficient of the quadratic equation to be solved can be arbitrarily set, 
    # and the setting of the initial value will affect the convergence rate. 
    initial_guess = [1, 1, 1]
    # Least squares method to solve the coefficient of the quadratic equation
    coefficients, cov = leastsq(error_curve, initial_guess, args=(x,y))
    # The coefficients of the quadratic equation are obtained.
    a, b, c = coefficients
    # Print the coefficients of the quadratic equation.
    print("curve function: y =",a," * x**2 + ",b ," * x + ",c)
    # Return the coefficients of the quadratic equation for further plotting.
    return a, b, c


