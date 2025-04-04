import numpy as np

# Define transformation matrices
theta_y = -np.pi / 6  # Rotation about Y-axis
theta_x = np.pi / 4   # Rotation about X-axis

# Rotation about Y-axis
R_y = np.array([
    [np.cos(theta_y), 0, np.sin(theta_y), 0],
    [0, 1, 0, 0],
    [-np.sin(theta_y), 0, np.cos(theta_y), 0],
    [0, 0, 0, 1]
])

# Rotation about X-axis
R_x = np.array([
    [1, 0, 0, 0],
    [0, np.cos(theta_x), -np.sin(theta_x), 0],
    [0, np.sin(theta_x), np.cos(theta_x), 0],
    [0, 0, 0, 1]
])

# Reflection across XZ-plane
R_xz = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# Translation matrix
T = np.array([
    [1, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 1, -2],
    [0, 0, 0, 1]
])

R = R_x @ R_y 
R = R[:3, :3]  # Extract the rotation part of the transformation matrix
print(R)  # Print the rotation matrix

v = np.array([3, -1, 4, 1])  # Homogeneous coordinates
# origin
o = np.array([0, 0, 0, 1])  # Homogeneous coordinates

# Compute final transformation matrix M
M = T @ R_xz @ R_x @ R_y
print(M)
print(M @ v)  # Apply transformation to vector v
print(M @ o)  # Apply transformation to origin o

import numpy as np

# Compute the numerical value of the rotation angle formula
theta = np.arccos((np.trace(R) - 1) / 2)

# Convert to degrees for better interpretability
theta_degrees = np.degrees(theta)
theta, theta_degrees
print(theta, theta_degrees)  # Print the angle in radians and degrees


n = (1/(2 * np.sin(theta))) * np.array([
    R[2,1] - R[1, 2],
    R[0, 2] - R[2, 0],
    R[1, 0] - R[0, 1]
])

n1, n2, n3 = n

print(n)  # Print the axis of rotation

N = np.array([
    [0, -n3, n2],
    [n3, 0, -n1],
    [-n2, n1, 0]
])

I = np.eye(3)  # Identity matrix

Rod = I + (1 - np.cos(theta)) * (N @ N) + np.sin(theta) * N
print(Rod)  # Print the Rodrigues rotation matrix


