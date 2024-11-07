import numpy as np
import matplotlib.pyplot as plt

class ExtendedKalmanFilter2D:
    def __init__(self, dt, process_variance, measurement_variance, landmarks):
        self.dt = dt
        self.landmarks = np.array(landmarks)

        # State vector [x, y, v_x, v_y]
        self.x = np.zeros((4, 1))

        # State transition matrix
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Process covariance matrix
        self.Q = process_variance * np.eye(4)

        # Measurement covariance matrix
        self.R = measurement_variance * np.eye(2 + len(landmarks))

        # Estimate covariance matrix
        self.P = np.eye(4)

    def predict(self):
        # Predict the state
        self.x = np.dot(self.F, self.x)

        # Predict the estimate covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        # Non-linear measurement function
        hx = np.array([
            [self.x[2, 0]],  # v_x
            [self.x[3, 0]],  # v_y
        ] + [
            [np.sqrt((self.x[0, 0] - lm[0])**2 + (self.x[1, 0] - lm[1])**2)] for lm in self.landmarks
        ])

        # Jacobian of the measurement function
        H = np.zeros((2 + len(self.landmarks), 4))
        H[0, 2] = 1  # Partial derivative of v_x
        H[1, 3] = 1  # Partial derivative of v_y

        for i, lm in enumerate(self.landmarks):
            range_ = np.sqrt((self.x[0, 0] - lm[0])**2 + (self.x[1, 0] - lm[1])**2)
            H[2 + i, 0] = (self.x[0, 0] - lm[0]) / range_
            H[2 + i, 1] = (self.x[1, 0] - lm[1]) / range_

        # Measurement residual
        y = z - hx

        # Residual covariance
        S = np.dot(H, np.dot(self.P, H.T)) + self.R

        # Kalman gain
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        # Update the state
        self.x = self.x + np.dot(K, y)

        # Update the estimate covariance
        I = np.eye(self.P.shape[0])
        self.P = (I - np.dot(K, H)).dot(self.P)

    def get_state(self):
        return self.x

# Simulation parameters
dt = 1.0  # Time step
process_variance = 1e-5  # Process noise variance
measurement_variance = 0.1  # Measurement noise variance
num_steps = 50
landmarks = [[2, 5], [10, 20], [15, 5], [30, 40], [35, 50], [45, 15]]  # Fixed landmark positions

# True trajectory (straight line with constant velocity)
true_positions = np.array([[i, i] for i in range(num_steps)])
true_velocities = np.array([[1, 1] for _ in range(num_steps)])

# Simulated noisy measurements
measurements = []
for pos, vel in zip(true_positions, true_velocities):
    distances = [np.sqrt((pos[0] - lm[0])**2 + (pos[1] - lm[1])**2) for lm in landmarks]
    noisy_measurement = np.array([
        vel[0] + np.random.normal(0, np.sqrt(measurement_variance)),
        vel[1] + np.random.normal(0, np.sqrt(measurement_variance)),
    ] + [
        d + np.random.normal(0, np.sqrt(measurement_variance)) for d in distances
    ])
    measurements.append(noisy_measurement)

# Initialize EKF
ekf = ExtendedKalmanFilter2D(dt, process_variance, measurement_variance, landmarks)

# Store estimated positions for plotting
estimated_positions = []

for z in measurements:
    ekf.predict()
    ekf.update(z.reshape(2 + len(landmarks), 1))
    estimated_state = ekf.get_state()
    estimated_positions.append(estimated_state[:2].flatten())

estimated_positions = np.array(estimated_positions)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(true_positions[:, 0], true_positions[:, 1], label='True Trajectory', linestyle='--', color='g')
plt.scatter([lm[0] for lm in landmarks], [lm[1] for lm in landmarks], label='Landmarks', color='k', marker='o')
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], label='Estimated Trajectory', color='b')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('2D Extended Kalman Filter with Multiple Landmarks')
plt.legend()
plt.grid()
plt.show()
