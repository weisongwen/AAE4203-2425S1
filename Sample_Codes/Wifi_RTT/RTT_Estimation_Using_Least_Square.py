import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the coordinates of the Access Points (APs).
ap_coords = np.array([
    [0, 0, 0],   # AP1 coordinates
    [6, 0, 0],   # AP2 coordinates
    [0, 8, 0],   # AP3 coordinates
    [0, 0, 7.5]  # AP4 coordinates
])

# The actual distances from the user to each AP.
dd = np.array([11, 9, 7, 13.5])

# List to store intermediate results
intermediate_results = []

# Define the residual function that calculates the difference between
# the predicted distances (from the guessed coordinates) and the measured distances.
def residuals(params):
    x, y, z, err = params   # x, y, z are the user's coordinates and err is an additional error term
    res = []   # List to hold the residuals for each AP
    for i in range(len(ap_coords)):
        xi, yi, zi = ap_coords[i]   # Coordinates of the i-th AP
        # Calculate the squared predicted distance from the guessed user position to the i-th AP
        predicted_d2 = (x - xi)**2 + (y - yi)**2 + (z - zi)**2
        # Adjust the measured distance by the error term and then square it
        measured_d2 = (dd[i] - err)**2
        # Calculate the residual (difference) and append to the list
        res.append(predicted_d2 - measured_d2)
    intermediate_results.append(params.copy())
    return res

# Initial guess for the user's position and the error term.
initial_guess = [10, 0, 3, 1]

# Solving the least squares problem
result = least_squares(residuals, initial_guess)

# Extract the optimized values from the result.
x, y, z, err = result.x
print(f"User coordinates: (x: {x:.2f}, y: {y:.2f}, z: {z:.2f})")
print(f"Error: {err:.2f}")

# Convert intermediate results to numpy array for easier indexing
intermediate_results = np.array(intermediate_results)

# Plotting setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the Access Points
ax.scatter(ap_coords[:, 0], ap_coords[:, 1], ap_coords[:, 2], c='r', marker='o', label='APs')

# Initialization function for animation
def init():
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    return []

# Update function for animation
def update(frame):
    ax.cla()  # Clear the axis
    ax.scatter(ap_coords[:, 0], ap_coords[:, 1], ap_coords[:, 2], c='r', marker='o', label='APs')
    ax.plot(intermediate_results[:frame+1, 0], intermediate_results[:frame+1, 1], intermediate_results[:frame+1, 2], color='grey', linestyle="--", label='Optimization Path')
    ax.scatter(intermediate_results[frame, 0], intermediate_results[frame, 1], intermediate_results[frame, 2], color='grey', marker='o')
    ax.scatter([x], [y], [z], c='g', marker='x', label='Final Estimated User Position')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    return []

# Create the animation
anim = FuncAnimation(fig, update, frames=len(intermediate_results), init_func=init, blit=False, repeat=False)
anim.save('rtt_ls.gif', writer='pillow')
