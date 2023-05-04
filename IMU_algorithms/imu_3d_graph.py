# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt
# import numpy as np


# def rotate_points(X, Y, Z, elev, azim, roll):
#     # Convert angles to radians
#     elev = np.deg2rad(elev)
#     azim = np.deg2rad(azim)
#     roll = np.deg2rad(roll)

#     # Create rotation matrices
#     Rx = np.array([[1, 0, 0],
#                    [0, np.cos(elev), -np.sin(elev)],
#                    [0, np.sin(elev), np.cos(elev)]])

#     Ry = np.array([[np.cos(azim), 0, np.sin(azim)],
#                    [0, 1, 0],
#                    [-np.sin(azim), 0, np.cos(azim)]])

#     Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
#                    [np.sin(roll), np.cos(roll), 0],
#                    [0, 0, 1]])

#     # Rotate points
#     R = np.dot(Rz, np.dot(Ry, Rx))
#     points = np.dot(R, np.vstack([X.ravel(), Y.ravel(), Z.ravel()]))

#     # Reshape rotated points to original shape
#     X_rot = points[0].reshape(X.shape)
#     Y_rot = points[1].reshape(Y.shape)
#     Z_rot = points[2].reshape(Z.shape)

#     return X_rot, Y_rot, Z_rot


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Grab some example data
# X, Y, Z = axes3d.get_test_data(0.05)

# print(X.shape)


# # Set the axis labels
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# line = np.linspace(0, 5, 1000)
# zeros_line = np.zeros(
#     len(line))

# ax.plot(line, zeros_line, zeros_line, color='red', linewidth=2)
# ax.plot(zeros_line, line, zeros_line, color='blue', linewidth=2)
# ax.plot(zeros_line, zeros_line, line, color='green', linewidth=2)
# # ax.plot(Y_rot[0], Y_rot[1], Y_rot[2], color='blue', linewidth=2)
# # ax.plot(Z_rot[0], Z_rot[1], Z_rot[2], color='green', linewidth=2)


# for angle in range(0, 360 * 4 + 1):
#     # Normalize the angle to the range [-180, 180] for display
#     angle_norm = (angle + 180) % 360 - 180

#     # Cycle through a full rotation of elevation, then azimuth, roll, and all
#     elev = azim = roll = 0
#     if angle <= 360:
#         elev = angle_norm
#     elif angle <= 360 * 2:
#         azim = angle_norm
#     elif angle <= 360 * 3:
#         roll = angle_norm
#     else:
#         elev = azim = roll = angle_norm

#     # Update the axis view
#     ax.view_init(elev, azim)

#     # Rotate points and update the wireframe
#     # X_rot, Y_rot, Z_rot = rotate_points(X, Y, Z, elev, azim, roll)
#     # ax.clear()
#     # ax.plot_wireframe(X_rot, Y_rot, Z_rot, rstride=10, cstride=10)

#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     plt.title('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (elev, azim, roll))

#     plt.draw()
#     plt.pause(0.001)

# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import time

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Make the X, Y meshgrid.
xs = np.linspace(-1, 1, 50)
ys = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(xs, ys)

# Set the z axis limits, so they aren't recalculated each frame.
ax.set_zlim(-1, 1)

# Begin plotting.
wframe = []
tstart = time.time()
for angle in range(0, 360):
    # If a line collection is already remove it before drawing.
    if wframe:
        for line in wframe:
            line.remove()

    # Generate data.
    phi = angle * np.pi / 180
    Z = np.cos(2 * np.pi * X + phi) * (1 - np.hypot(X, Y))

    # Update the view angle
    ax.view_init(azim=angle)

    # Plot the new wireframe and pause briefly before continuing.
    wframe = [ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)]

    # Add the 3D axis lines to wframe
    line = np.array([1, 0, 0])
    zeros_line = np.array([0, 0, 0])
    wframe.append(ax.plot(line, zeros_line, zeros_line,
                  color='red', linewidth=2)[0])
    wframe.append(ax.plot(zeros_line, line, zeros_line,
                  color='blue', linewidth=2)[0])
    wframe.append(ax.plot(zeros_line, zeros_line,
                  line, color='green', linewidth=2)[0])

    plt.draw()
    plt.pause(.001)

print('Average FPS: %f' % (360 / (time.time() - tstart)))
