import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from madgwick_py.madgwickahrs import MadgwickAHRS
import numpy as np

column_names = ['Sampled Time (ms)', 'Encoder (deg)',
                'IMU2q0', 'IMU2q1', 'IMU2q2', 'IMU2q3',
                'IMU1q0', 'IMU1q1', 'IMU1q2', 'IMU1q3',
                'IMU2GyroX', 'IMU2GyroY', 'IMU2GyroZ',
                'IMU2AccelX', 'IMU2AccelY', 'IMU2AccelZ',
                'IMU1GyroX', 'IMU1GyroY', 'IMU1GyroZ',
                'IMU1AccelX', 'IMU1AccelY', 'IMU1AccelZ']

df = pd.read_csv(
    '../raw_processed_IMU_data/raw_pitch_fast.txt', names=column_names)

# Read in the text file
with open("../raw_processed_IMU_data/raw_pitch_fast.txt") as file:
    text = file.readlines()

# Find the index where the marker occurs
marker_index = next((i for i, line in enumerate(text)
                    if "Recording IMU" in line), None)

# Split the text into two dataframes based on the marker
calibration_data = pd.read_csv('../raw_processed_IMU_data/raw_pitch_fast.txt',
                               names=column_names, nrows=marker_index, header=None)


imu_data = pd.read_csv('../raw_processed_IMU_data/raw_pitch_fast.txt',
                       names=column_names, skiprows=marker_index, header=None)
imu_data = imu_data[1:-4]
# imu_data = pd.read_csv("\n".join(text[marker_index:]), header=None)

# imu_data['IMU2GyroZ'][:1000].plot()
# plt.title('Plot of column values')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.show()

# imu_data['IMU2AccelX'][:1000].plot()
# plt.title('Plot of column values')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.show()

sample_period = 1.0 / 100  # Sample period of 1/256 seconds
beta = 1  # Default gain value
madgwick_custom = MadgwickAHRS(sampleperiod=sample_period, beta=beta)

z_axis = []
vec = []
gyro = []

rotated_v = np.array([1, 0, 0])

for index, row in imu_data.iterrows():
    # Extract gyroscope, accelerometer and magnetometer values from each row
    gyroscope = [row['IMU2GyroX'], row['IMU2GyroY'], row['IMU2GyroZ']]
    accelerometer = [row['IMU2AccelX'], row['IMU2AccelY'], row['IMU2AccelZ']]
    #     magnetometer = [row[7], row[8], row[9]]
    #     print(gyroscope)
    madgwick_custom.update_imu(gyroscope, accelerometer)
    z_rot = madgwick_custom.quaternion.to_z_rotation()

    roll, pitch, yaw = madgwick_custom.quaternion.to_euler_angles()

    # # vec.append(rotated_v)

    # # create a rotation matrix for the z-axis rotation
    # R = np.array([[np.cos(rad), -np.sin(rad), 0],
    #               [np.sin(rad), np.cos(rad), 0],
    #               [0, 0, 1]])

    # # rotate the vector using the rotation matrix
    # rotated_v = R @ rotated_v

    z_axis.append(z_rot)

    # z_axis.append(yaw)
    # print(z_axis)

    gyro.append(gyroscope[1])

    if index >= 100:
        break

print(z_axis)


plt.plot(z_axis)
imu_data['Encoder (deg)'][:100].plot()
# plt.title('Plot of column values')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.show()


# plt.plot(gyro)
plt.title('Plot of column values')
# plt.xlabel('Index')
# plt.ylabel('Value')
plt.show()
