import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具

# Read the data from CSV files
satellite_positions = np.loadtxt('E:/TA/rtklib_matlab/satellite_positions.csv', delimiter=',')  # (max_num_sats, num_epochs*3)
pseudoranges_meas = np.loadtxt('E:/TA/rtklib_matlab/pseudoranges_meas.csv', delimiter=',')  # (max_num_sats, num_epochs)

satellite_clock_bias = np.loadtxt('E:/TA/rtklib_matlab/satellite_clock_bias.csv', delimiter=',')  # (max_num_sats, num_epochs)
ionospheric_delay = np.loadtxt('E:/TA/rtklib_matlab/ionospheric_delay.csv', delimiter=',')  # (max_num_sats, num_epochs)
tropospheric_delay = np.loadtxt('E:/TA/rtklib_matlab/tropospheric_delay.csv', delimiter=',')  # (max_num_sats, num_epochs)

# Get the number of epochs
num_epochs = pseudoranges_meas.shape[1]
max_num_sats = pseudoranges_meas.shape[0]

# Initialize variables to store estimated positions
estimated_positions = []  # List to store estimated receiver positions
estimated_clock_biases = []  # List to store estimated receiver clock biases

# Set initial receiver position (you can set this to a known approximate position)
receiver_position = np.array([0.0, 0.0, 0.0])  # Units: meters

c = 299792458.0  # Speed of light, meters per second

def least_squares_solution(satellite_positions, receiver_position, pseudoranges_meas, satellite_clock_bias,
                           ionospheric_delay, tropospheric_delay):
    receiver_clock_bias = 0.0  # Receiver clock bias, in meters
    for j in range(10):  # Maximum iterations
        # Compute geometric distances
        estimated_distances = np.linalg.norm(satellite_positions - receiver_position, axis=1)

        # Correct pseudorange measurements
        corrected_pseudoranges = pseudoranges_meas + satellite_clock_bias - ionospheric_delay - tropospheric_delay

        # Compute residuals
        pseudoranges_diff = corrected_pseudoranges - (estimated_distances + receiver_clock_bias)

        # Build the design matrix G
        G = np.zeros((len(satellite_positions), 4))
        for i in range(len(satellite_positions)):
            p_i = satellite_positions[i] - receiver_position
            r_i = estimated_distances[i]
            G[i, :3] = -p_i / r_i
            G[i, 3] = 1.0

        # Solve using least squares
        delta_p, residuals, rank, s = np.linalg.lstsq(G, pseudoranges_diff, rcond=None)
        receiver_position += delta_p[:3]
        receiver_clock_bias += delta_p[3]

        # Check convergence
        if np.linalg.norm(delta_p[:3]) < 1e-4:
            break
    return receiver_position, receiver_clock_bias

def ecef_to_lla(x, y, z):
    # WGS84 ellipsoid constants
    a = 6378137.0  # Semi-major axis, meters
    e2 = 6.69437999014e-3  # Square of eccentricity

    # Longitude calculation
    lon = np.arctan2(y, x)

    # Latitude and altitude initial estimation
    p = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p * (1 - e2))  # Initial latitude
    lat_prev = 0
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - N

    # Iterative computation
    while np.abs(lat - lat_prev) > 1e-12:
        lat_prev = lat
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))

    # Convert from radians to degrees
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)
    return lat_deg, lon_deg, h

def ecef_to_enu(x, y, z, x_ref, y_ref, z_ref):
    # Convert reference point to lat, lon, h
    lat_ref, lon_ref, h_ref = ecef_to_lla(x_ref, y_ref, z_ref)
    lat_ref = np.radians(lat_ref)
    lon_ref = np.radians(lon_ref)

    # Compute difference vector
    dx = x - x_ref
    dy = y - y_ref
    dz = z - z_ref

    # Transformation matrix
    t = np.array([
        [-np.sin(lon_ref),               np.cos(lon_ref),              0],
        [-np.sin(lat_ref)*np.cos(lon_ref), -np.sin(lat_ref)*np.sin(lon_ref), np.cos(lat_ref)],
        [np.cos(lat_ref)*np.cos(lon_ref),  np.cos(lat_ref)*np.sin(lon_ref), np.sin(lat_ref)]
    ])

    # Apply transformation
    enu = t @ np.array([dx, dy, dz])
    return enu

# Loop over all epochs
for epoch in range(num_epochs):
    # Extract data for the current epoch
    p_l1_epoch = pseudoranges_meas[:, epoch]
    sat_clock_err_epoch = satellite_clock_bias[:, epoch]
    ion_error_l1_epoch = ionospheric_delay[:, epoch]
    tropo_error_epoch = tropospheric_delay[:, epoch]
    sat_pos_epoch = satellite_positions[:, epoch*3:(epoch+1)*3]  # Columns for current epoch

    # Exclude NaN values
    valid_idx = ~np.isnan(p_l1_epoch) & \
                ~np.isnan(sat_clock_err_epoch) & \
                ~np.isnan(ion_error_l1_epoch) & \
                ~np.isnan(tropo_error_epoch) & \
                ~np.isnan(sat_pos_epoch[:, 0]) & \
                ~np.isnan(sat_pos_epoch[:, 1]) & \
                ~np.isnan(sat_pos_epoch[:, 2])

    # Check if enough satellites are available
    if np.sum(valid_idx) < 4:
        print(f"Epoch {epoch+1}: Not enough satellites, skipping this epoch.")
        if epoch > 0:
            # Use the previous position
            estimated_positions.append(estimated_positions[-1])
            estimated_clock_biases.append(estimated_clock_biases[-1])
        else:
            # Use the initial position
            estimated_positions.append(receiver_position.copy())
            estimated_clock_biases.append(0.0)
        continue

    # Extract valid data
    p_l1_valid = p_l1_epoch[valid_idx]
    sat_clock_err_valid = sat_clock_err_epoch[valid_idx]
    ion_error_l1_valid = ion_error_l1_epoch[valid_idx]
    tropo_error_valid = tropo_error_epoch[valid_idx]
    sat_pos_valid = sat_pos_epoch[valid_idx, :]

    # Use previous epoch's position as initial estimate
    if epoch > 0:
        receiver_position = estimated_positions[-1].copy()
    else:
        receiver_position = np.array([0.0, 0.0, 0.0])  # Initial position

    # Perform least squares estimation
    estimated_position, estimated_receiver_clock_bias = least_squares_solution(
        sat_pos_valid, receiver_position, p_l1_valid, sat_clock_err_valid, ion_error_l1_valid, tropo_error_valid
    )

    # Store the estimated position and clock bias
    estimated_positions.append(estimated_position.copy())
    estimated_clock_biases.append(estimated_receiver_clock_bias)

# Convert ECEF positions to latitude, longitude, and altitude
lat_list = []
lon_list = []
alt_list = []
for pos in estimated_positions:
    lat, lon, alt = ecef_to_lla(pos[0], pos[1], pos[2])
    lat_list.append(lat)
    lon_list.append(lon)
    alt_list.append(alt)

# Compute ENU coordinates with respect to the first estimated position
enu_list = []
x_ref, y_ref, z_ref = estimated_positions[0]  # Reference point
for pos in estimated_positions:
    enu = ecef_to_enu(pos[0], pos[1], pos[2], x_ref, y_ref, z_ref)
    enu_list.append(enu)

# Extract East, North, Up components
east_list = [enu[0] for enu in enu_list]
north_list = [enu[1] for enu in enu_list]
up_list = [enu[2] for enu in enu_list]

# Print the estimated positions for the last epoch
print(f"Estimated Position in Latitude, Longitude, Altitude for last epoch:")
print(f"Latitude: {lat_list[-1]} degrees")
print(f"Longitude: {lon_list[-1]} degrees")
print(f"Altitude: {alt_list[-1]} meters")

# Plot the receiver's trajectory in Latitude and Longitude
plt.figure()
plt.plot(lon_list, lat_list, 'r.-')
plt.title('Estimated Receiver Trajectory')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

# Plot the ENU trajectory in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(east_list, north_list, up_list, 'b.-')
ax.set_title('Receiver Trajectory in ENU Coordinates (3D)')
ax.set_xlabel('East (meters)')
ax.set_ylabel('North (meters)')
ax.set_zlabel('Up (meters)')
ax.grid(True)
plt.show()
