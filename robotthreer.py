import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la

# Link lengths
L1, L2, L3 = 5, 3, 2

# Joint angle limits (radians)
theta1_min, theta1_max = -np.pi, np.pi
theta2_min, theta2_max = -np.pi, np.pi
theta3_min, theta3_max = -np.pi, np.pi

# Sampling resolution
num_points = 30

# Generate joint angle vectors for workspace sampling
theta1 = np.linspace(theta1_min, theta1_max, num_points)
theta2 = np.linspace(theta2_min, theta2_max, num_points)
theta3 = np.linspace(theta3_min, theta3_max, num_points)

# Compute workspace points by forward kinematics
workspace_x, workspace_y = [], []
for t1 in theta1:
    for t2 in theta2:
        for t3 in theta3:
            x = L1*np.cos(t1) + L2*np.cos(t1 + t2) + L3*np.cos(t1 + t2 + t3)
            y = L1*np.sin(t1) + L2*np.sin(t1 + t2) + L3*np.sin(t1 + t2 + t3)
            workspace_x.append(x)
            workspace_y.append(y)

# Plot workspace
plt.figure()
plt.plot(workspace_x, workspace_y, '.', markersize=5)
plt.axis('equal')
plt.grid(True)
plt.title('Workspace of 3R Robot Arm')
plt.xlabel('X Position')
plt.ylabel('Y Position')

# Plot sample robot arm configuration
sample_theta = [np.pi/4, np.pi/6, -np.pi/6]
joint1 = np.array([0, 0])
joint2 = joint1 + np.array([L1*np.cos(sample_theta[0]), L1*np.sin(sample_theta[0])])
joint3 = joint2 + np.array([L2*np.cos(sample_theta[0]+sample_theta[1]), L2*np.sin(sample_theta[0]+sample_theta[1])])
end_effector = joint3 + np.array([L3*np.cos(sum(sample_theta)), L3*np.sin(sum(sample_theta))])

plt.plot([joint1[0], joint2[0], joint3[0], end_effector[0]], 
         [joint1[1], joint2[1], joint3[1], end_effector[1]], 'r-', linewidth=2)
plt.plot(end_effector[0], end_effector[1], 'ro', markersize=8, markerfacecolor='r')
plt.legend(['Workspace Points', '3R Arm Sample Configuration'])

# Plot Configuration Space (C-Space)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Theta1, Theta2, Theta3 = np.meshgrid(theta1, theta2, theta3)
ax.scatter(Theta1.flatten(), Theta2.flatten(), Theta3.flatten(), c='g', marker='.')
ax.grid(True)
ax.set_title('Configuration Space (C-Space) of 3R Robot Arm')
ax.set_xlabel(r'$\theta_1$ (rad)')
ax.set_ylabel(r'$\theta_2$ (rad)')
ax.set_zlabel(r'$\theta_3$ (rad)')
ax.set_xlim([theta1_min, theta1_max])
ax.set_ylim([theta2_min, theta2_max])
ax.set_zlim([theta3_min, theta3_max])

# Jacobian matrix at sample configuration
theta1, theta2, theta3 = sample_theta
J = np.zeros((2, 3))
J[:,0] = [-L1*np.sin(theta1) - L2*np.sin(theta1 + theta2) - L3*np.sin(theta1 + theta2 + theta3),
          L1*np.cos(theta1) + L2*np.cos(theta1 + theta2) + L3*np.cos(theta1 + theta2 + theta3)]
J[:,1] = [-L2*np.sin(theta1 + theta2) - L3*np.sin(theta1 + theta2 + theta3),
          L2*np.cos(theta1 + theta2) + L3*np.cos(theta1 + theta2 + theta3)]
J[:,2] = [-L3*np.sin(theta1 + theta2 + theta3),
          L3*np.cos(theta1 + theta2 + theta3)]

# Compute null space basis vectors of Jacobian
null_vecs = la.null_space(J)

# Plot null space vectors in joint velocity space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0, 0, 0, null_vecs[0,0], null_vecs[1,0], null_vecs[2,0], color='r', linewidth=2)
ax.quiver(0, 0, 0, -null_vecs[0,0], -null_vecs[1,0], -null_vecs[2,0], color='r', linewidth=2)
ax.grid(True)
ax.set_xlabel('Joint 1 velocity')
ax.set_ylabel('Joint 2 velocity')
ax.set_zlabel('Joint 3 velocity')
ax.set_title('Null Space of 3R Robot Arm Jacobian at Sample Configuration')
ax.set_box_aspect([1,1,1])

plt.show()

# Display null space vectors
print('Null space vectors (joint velocities causing no end-effector motion):')
print(null_vecs)
