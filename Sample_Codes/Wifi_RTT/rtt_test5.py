import numpy as np
from scipy.optimize import least_squares

# Define the coordinates of the Access Points (APs).
ap_coords = np.array([
    [0, 0, 0],   # AP1 coordinates
    [6, 0, 0],   # AP2 coordinates
    [0, 8, 0],   # AP3 coordinates
    [0, 0, 7.5] # AP4 coordinates, possibly located below the level of the others
])

# The actual distances from the user to each AP.
dd = np.array([11, 9, 7, 13.5])

# Define the residual function that calculates the difference between
# the predicted distances (from the guessed coordinates) and the measured distances.
def residuals(params):
    x, y, z, err = params  # x, y, z are the user's coordinates and err is an additional error term
    res = []  # List to hold the residuals for each AP
    for i in range(len(ap_coords)):
        xi, yi, zi = ap_coords[i]  # Coordinates of the i-th AP
        # Calculate the squared predicted distance from the guessed user position to the i-th AP
        predicted_d2 = (x - xi)**2 + (y - yi)**2 + (z - zi)**2
        # Adjust the measured distance by the error term and then square it
        measured_d2 = (dd[i] - err)**2
        # Calculate the residual (difference) and append to the list
        res.append(predicted_d2 - measured_d2)
    return res

# Initial guess for the user's position and the error term.
initial_guess = [0, 0, 0, 1]

# Solving the least squares problem using the residuals function and the initial guess.
result = least_squares(residuals, initial_guess)

# Extract the optimized values from the result.
x, y, z, err = result.x
print(f"User coordinates: (x: {x:.2f}, y: {y:.2f}, z: {z:.2f})")
print(f"Error: {err:.2f}")
