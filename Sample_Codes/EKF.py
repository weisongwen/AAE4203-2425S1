import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter2D:
    def __init__(self, dt, process_variance, measurement_variance):
        # Time step
        self.dt = dt

        # State vector [x_position, x_velocity, y_position, y_velocity]
        self.x = np.zeros((4, 1))

        # State transition matrix
        self.F = np.array([[1, dt, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, dt],
                           [0, 0, 0, 1]])

        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])

        # Process covariance matrix
        self.Q = process_variance * np.eye(4)

        # Measurement covariance matrix
        self.R = measurement_variance * np.eye(2)

        # Estimate covariance matrix
        self.P = np.eye(4)

    def predict(self):
        # Predict the state
        self.x = np.dot(self.F, self.x)

        # Predict the estimate covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        # Measurement residual
        y = z - np.dot(self.H, self.x)

        # Residual covariance
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update the state
        self.x = self.x + np.dot(K, y)

        # Update the estimate covariance
        I = np.eye(self.H.shape[1])
        self.P = (I - np.dot(K, self.H)).dot(self.P)

    def get_state(self):
        return self.x

# Simulation parameters
dt = 1.0  # Time step
process_variance = 1e-5  # Process noise variance
measurement_variance = 0.1  # Measurement noise variance
num_steps = 50

# True trajectory (straight line with constant velocity)
true_positions = np.array([[i, i] for i in range(num_steps)])

# Simulated noisy measurements
measurements = true_positions + np.random.normal(0, np.sqrt(measurement_variance), true_positions.shape)

# Initialize Kalman filter
kf = KalmanFilter2D(dt, process_variance, measurement_variance)

# Store estimated positions for plotting
estimated_positions = []

for z in measurements:
    kf.predict()
    kf.update(z.reshape(2, 1))
    estimated_state = kf.get_state()
    estimated_positions.append(estimated_state[[0, 2]].flatten())

estimated_positions = np.array(estimated_positions)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(true_positions[:, 0], true_positions[:, 1], label='True Trajectory', linestyle='--', color='g')
plt.scatter(measurements[:, 0], measurements[:, 1], label='Measurements', color='r', marker='x')
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], label='Estimated Trajectory', color='b')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('2D Kalman Filter Position Estimation')
plt.legend()
plt.grid()
plt.show()
