import numpy as np
import matplotlib.pyplot as plt

# Link lengths
L1 = 20
L2 = 25

# Joint angle limits in degrees
theta1_min, theta1_max = 0, 360
theta2_min, theta2_max = -180, 180

# Convert to radians
theta1_vals = np.radians(np.linspace(theta1_min, theta1_max, 120))
theta2_vals = np.radians(np.linspace(theta2_min, theta2_max, 120))

# Lists for storing workspace points
workspace_x = []
workspace_y = []

for t1 in theta1_vals:
    for t2 in theta2_vals:
        # Forward Kinematics using transformation matrices
        x = L1*np.cos(t1) + L2*np.cos(t1 + t2)
        y = L1*np.sin(t1) + L2*np.sin(t1 + t2)
        
        workspace_x.append(x)
        workspace_y.append(y)

# Example Configuration: θ1 = 45°, θ2 = 30°
t1_sample = np.radians(45)
t2_sample = np.radians(30)

x1 = L1*np.cos(t1_sample)
y1 = L1*np.sin(t1_sample)

x2 = x1 + L2*np.cos(t1_sample + t2_sample)
y2 = y1 + L2*np.sin(t1_sample + t2_sample)

# Plotting
plt.figure(figsize=(7, 7))
plt.scatter(workspace_x, workspace_y, s=3, color='blue', label='Workspace Points')

# Plot robot arm configuration
plt.plot([0, x1, x2], [0, y1, y2], '-o', color='red', linewidth=2, markersize=8, label='2R Arm Configuration')

plt.title("Workspace of 2R Robot Arm")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()