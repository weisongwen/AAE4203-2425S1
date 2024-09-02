import numpy as np

# Import the satellite position Unit: meter
satellite_positions = np.array([
    [-13186870.6, 11385729.2, 19672626.3],
    [-7118031.6, 23256076.0, -9700477.9],
    [-2303925.9, 17164155.9, 20120354.5],
    [-15426414.5, 2696509.3, 22137570.3]
])

# Import the pseudoranges measurement Unit: meter
pseudoranges_meas = np.array([21196662.1, 22222028.54, 21431397.16, 23928467.12])

# Import the satellite clock bias(δ_(𝑟,𝑡)^𝑠) Unit: meter
satellite_clock_bias = np.array([198812.8, 52245.17, 21575.56, 37173.51])

# Import the ionospheric delay Unit: meter
ionospheric_delay = np.array([3.8639, 4.6762, 3.614, 5.9277])

# Import the tropospheric delay Unit: meter
tropospheric_delay = np.array([3.24, 4.32, 3.07, 5.60])

# Set initial receiver position
receiver_position = np.array([0.0, 0.0, 0.0])



"""Calculate solution of receiver position by least squares, iterate a maximum of 20 times until the condition is met
Parameters:
satellite_positions - A three-dimensional array of satellite positions
receiver_position -The receiver positions
satellite_clock_bias - The value of the satellite clock bias 
ionospheric_delay - The value of the ionospheric delay
tropospheric_delay - The value of tropospheric delay """
def least_squares_solution(satellite_positions, receiver_position, pseudoranges_meas,satellite_clock_bias, ionospheric_delay, tropospheric_delay):
    for j in range(20):
        # Calculate the pseudorange
        estimated_distances = np.linalg.norm(satellite_positions - receiver_position, axis=1)
        pseudoranges = estimated_distances-satellite_clock_bias+ionospheric_delay+tropospheric_delay
        # Calculate the difference between the measured pseudorange and the calculated pseudorange
        pseudoranges_diff = pseudoranges_meas - pseudoranges

        # Initialize the matrix G
        G = np.zeros((len(satellite_positions), 4))
        # Calculate the matrix G
        for i in range(len(satellite_positions)):
            p_i = satellite_positions[i] - receiver_position
            r_i = np.linalg.norm(p_i)
            G[i, :3] = -p_i / r_i
            G[i, 3] = 1.0

        # Solve using least square method
        #delta_p = np.linalg.inv(G.T @ G) @ G.T @ pseudoranges_diff
        delta_p = np.linalg.lstsq(G, pseudoranges_diff, rcond=None)[0]
        receiver_position += delta_p[:3]

        print(f"Iteration time =  {j}")
        print(f"pseudoranges =  {pseudoranges}")
        print(f"estimated_distances =  {estimated_distances}")
        print(f"pseudoranges_diff =  {pseudoranges_diff}")
        print(f"G =  {G}")
        print(f"delta_p =  {delta_p[:3]}")
        print(f"Estimated Receiver Position: {receiver_position}")
        print(f"Estimated Receiver Clock Bias: {delta_p[3]}")

        if np.linalg.norm(delta_p[:3]) < 1e-4:
            break
    return receiver_position


# Use the least square method to solve the receiver position
estimated_position = least_squares_solution(satellite_positions, receiver_position, pseudoranges_meas,satellite_clock_bias, ionospheric_delay, tropospheric_delay)

