import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import seaborn as sns

# from madgwick_py.madgwickahrs import MadgwickAHRS
from madgwick_py.quaternion import Quaternion
from ahrs.ahrs.filters.madgwick import Madgwick
from ahrs.ahrs.filters.mahony import Mahony
from ahrs.ahrs.filters.ekf import EKF

from imucal.management import load_calibration_info
from imucal import FerrarisCalibration
from imucal import ferraris_regions_from_df

import numpy as np
import time


import serial

from mpl_toolkits.mplot3d import axes3d

import time

sns.set_style("darkgrid")
sns.set(rc={"figure.figsize": (12, 4)})


def euler_values_ahrs(custom_function, Q_prev, time_prev, data):

    # print(data)

    # Extract gyroscope, accelerometer and magnetometer values from each row
    gyroscope = [data['gyr_x'][0],
                 data['gyr_y'][0], data['gyr_z'][0]]
    accelerometer = [data['acc_x'][0]*9.806,
                     data['acc_y'][0]*9.806, data['acc_z'][0]*9.806]

    time_curr = int(data["Sampled Time (ms)"])
    delta_t = (time_curr - time_prev) / 1000

    # print(Q_prev)
    # print("gyroscope", gyroscope)
    # print("accelerometer", accelerometer)
    # print("delta_t", delta_t)

    Q = custom_function.updateIMU(
        q=Q_prev, gyr=gyroscope, acc=accelerometer, dt=float(delta_t))

    Q_prev = Q

    quaternion = Quaternion(w_or_q=Q_prev)

    euler_angle = quaternion.to_euler_angles()

    return np.array(euler_angle), Q, time_curr


def calibration(df):

    sampling_rate = 1000 / (df["Sampled Time (ms)"]
                            [1] - df["Sampled Time (ms)"][0])  # Hz

    print("sampling_rate", sampling_rate)

    df = df.set_index(["part"])

    # cal_df = cal_df.reset_index()

    # print(df)

    regions = ferraris_regions_from_df(df)  # Pass cal_df instead of df
    # print(regions)

    cal = FerrarisCalibration()
    cal_mat = cal.compute(regions, sampling_rate,
                          from_acc_unit="m/s^2", from_gyr_unit="deg/s")

    # `cal_mat` is your final calibration matrix object you can use to calibrate data
    cal_mat.to_json_file('./calibration.json')

    return df


def calibration_df(calibration_data):

    df = pd.DataFrame(
        calibration_data,
        columns=["part", "IMUAccel", "IMUGyro_rad", "YPR", "Sampled Time (ms)"]
    )

    # Split the IMUGyro_rad and IMUAccel columns into separate columns
    df[['gyr_x', 'gyr_y', 'gyr_z']] = pd.DataFrame(
        df['IMUGyro_rad'].tolist(), index=df.index)
    df[['acc_x', 'acc_y', 'acc_z']] = pd.DataFrame(
        df['IMUAccel'].tolist(), index=df.index)
    df[['Y', 'P', 'R']] = pd.DataFrame(
        df['YPR'].tolist(), index=df.index)

    # Drop the original IMUGyro_rad and IMUAccel columns
    df.drop(['IMUGyro_rad', 'IMUAccel', 'YPR'], axis=1, inplace=True)

    # Reorder columns
    df = df[["part", 'acc_x', 'acc_y', 'acc_z', 'gyr_x',
            'gyr_y', 'gyr_z', 'Y', 'P', 'R', 'Sampled Time (ms)']]

    df_uncal = df.copy()

    return calibration(df), df_uncal


def data_df(data):
    data_dict = {'part': data[0],
                 'acc_x': data[1][0],
                 'acc_y': data[1][1],
                 'acc_z': data[1][2],
                 'gyr_x': data[2][0],
                 'gyr_y': data[2][1],
                 'gyr_z': data[2][2],
                 'Y': data[3][0],
                 'P': data[3][1],
                 'R': data[3][2],
                 'Sampled Time (ms)': data[4]}

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([data_dict])

    df = df.set_index(["part"])

    # print(df)

    return df


def parse_line(line, data_obj, capture):
    decoded_line = line.decode('utf-8').strip()
    # decoded_line = line[2:-6].strip()

    labels = ['data', 'z_p', 'z_a', 'x_p', 'x_a',
              'y_p', 'y_a', 'z_rot', 'x_rot', 'y_rot']

    # print(decoded_line)

    if "booted" in decoded_line:
        return -1, False

    try:
        if decoded_line in labels:
            capture = True
            data_obj = [decoded_line]
        elif ',' in decoded_line and capture:
            data_obj.append(list(map(float, decoded_line.split(', '))))
        elif capture:
            data_obj.append(int(decoded_line))
    except:
        data_obj = []
        capture = False

    return data_obj, capture


def write_loop():

    # Open the serial device
    ser = serial.Serial('/dev/tty.usbserial-120', 115200)

    # Open the output text file
    with open('output.txt', 'w') as f:
        while True:
            # Read a line from the serial device
            line = ser.readline()
            
            # Convert the line from bytes to a string
            # line_str = line.decode('utf-8')

            # Print the line
            print(str(line))

            # Write the line to the file
            f.write(str(line))


def main_loop():

    ser = serial.Serial('/dev/tty.usbserial-120', 115200)

    # file_imu = open('new_cal_data.txt', 'r')
    # Lines = file_imu.readlines()

    # file_imu.close()

    cal_labels = ['z_p', 'z_a', 'x_p', 'x_a',
                  'y_p', 'y_a', 'z_rot', 'x_rot', 'y_rot']

    data_obj = []
    calibration_data = []
    capture = False
    cal_complete = False
    # cal_mat = None

    euler_angles = []

    madgwick_ahrs = EKF()
    cal_complete = True

    # Begin plotting.
    wframe = []
    # tstart = time.time()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Set the z axis limits, so they aren't recalculated each frame.
    ax.set_zlim(-1, 1)


    # Grab some example data and plot a basic wireframe.
    # X, Y, Z = axes3d.get_test_data(0.05)
    # ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    # # Define the axis vectors
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    # Plot the rotated axis vectors
    origin = np.array([0, 0, 0])

    ax.plot(*zip(origin, x_axis),
                      color='red', linewidth=2)[0]
    ax.plot(*zip(origin, y_axis),
                          color='blue', linewidth=2)[0]
    ax.plot(*zip(origin, z_axis),
                          color='green', linewidth=2)[0]

    ax.set_axis_off()

    # Set the axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    Q_prev = np.array([1,0,0,0])
    time_prev = 0

    cal_mat = load_calibration_info('./calibration.json')

    while True:
        start_time = time.process_time()

        line = ser.readline()
        # print(line)

        data_obj, capture = parse_line(line, data_obj, capture)

        if data_obj == -1:
            data_obj = []
            calibration_data = []
            cal_complete = False
            print("RESET!!")

        # print(data_obj)

       

        if len(data_obj) == 5 and data_obj[0] == "reset":
            calibration_data = []
        elif len(data_obj) == 5 and data_obj[0] == "data" and cal_complete:

            capture = False

            # if cal_mat == None:
            #     print("Not Calibrated")
            #     continue

            data_cal_df = data_df(data_obj)

            print(data_cal_df)

            data = cal_mat.calibrate_df(
                data_cal_df, acc_unit="m/s^2", gyr_unit="deg/s")

            # print(row['acc_x'])
            euler_angle, Q_prev, time_prev = euler_values_ahrs(
                madgwick_ahrs, Q_prev, time_prev, data_cal_df)

            print(time.process_time() - start_time)

            # euler_angles.append(np.array(euler_angle))

            # if wframe:
            #     for line in wframe:
            #         line.remove()

            # # Create a rotation object from Euler angles
            # rot = Rotation.from_euler(
            #     'xyz', [data_cal_df['Y'][0], data_cal_df['P'][0], data_cal_df['R'][0]])

            # # Define the axis vectors
            # x_axis = np.array([1, 0, 0])
            # y_axis = np.array([0, 1, 0])
            # z_axis = np.array([0, 0, 1])

            # # Rotate the axis vectors
            # x_rotated = rot.apply(x_axis)
            # y_rotated = rot.apply(y_axis)
            # z_rotated = rot.apply(z_axis)

            # # Plot the rotated axis vectors
            # origin = np.array([0, 0, 0])

            # wframe = [ax.plot(*zip(origin, x_rotated),
            #                   color='red', linewidth=2)[0]]
            # wframe.append(ax.plot(*zip(origin, y_rotated),
            #                       color='blue', linewidth=2)[0])
            # wframe.append(ax.plot(*zip(origin, z_rotated),
            #                       color='green', linewidth=2)[0])

            # plt.draw()

            # plt.pause(.001)


            # Normalize the angle to the range [-180, 180] for display
            # angle_norm = (angle + 180) % 360 - 180

            # # Cycle through a full rotation of elevation, then azimuth, roll, and all
            # elev = azim = roll = 0
            # if angle <= 360:
            #     elev = angle_norm
            # elif angle <= 360*2:
            #     azim = angle_norm
            # elif angle <= 360*3:
            #     roll = angle_norm
            # else:
            #     elev = azim = roll = angle_norm

            # print('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (data_cal_df['Y'], data_cal_df['P'], data_cal_df['R']))

            # Update the axis view and title
            ax.view_init(data_cal_df['R'][0], data_cal_df['Y'][0])
            plt.title('Pitch: %d°, Yaw: %d°, Roll: %d°' % (data_cal_df['R'], data_cal_df['Y'],data_cal_df['P']))

            plt.draw()
            plt.pause(.001)

            

        elif len(data_obj) == 5 and data_obj[0] == "data":
            print("CALLL COMPLETED!!!!!!!")
            cal_complete = True
            cal_df, uncal_df = calibration_df(calibration_data)

            cal_mat = load_calibration_info('./calibration.json')
            calibrated_data = cal_mat.calibrate_df(
                cal_df, acc_unit="m/s^2", gyr_unit="deg/s")

            calibrated_data = calibrated_data.reset_index()
            cal_df = cal_df.reset_index()
            # calibrated_data = calibrated_data.drop(['part'], axis=1, inplace=True)

            Q_prev = np.array([1., 0., 0., 0.])
            time_prev = calibrated_data["Sampled Time (ms)"][0] - (
                calibrated_data["Sampled Time (ms)"][1] - calibrated_data["Sampled Time (ms)"][0])

            # Plot acceleration data
            plt.figure(1)
            plt.figure(figsize=(8, 4))
            plt.plot(calibrated_data['acc_x'], label='acc_x')
            plt.plot(calibrated_data['acc_y'], label='acc_y')
            plt.plot(calibrated_data['acc_z'], label='acc_z')
            plt.xlabel('Sampled Time (ms)')
            plt.ylabel('Acceleration (g)')
            plt.title('Calibrated Acceleration Data')
            plt.legend()

            plt.figure(2)
            plt.figure(figsize=(8, 4))
            plt.plot(uncal_df['acc_x'], label='acc_x')
            plt.plot(uncal_df['acc_y'], label='acc_y')
            plt.plot(uncal_df['acc_z'], label='acc_z')
            plt.xlabel('Sampled Time (ms)')
            plt.ylabel('Acceleration (g)')
            plt.title('Uncalibrated Acceleration Data')
            plt.legend()

            plt.figure(3)
            # Plot gyroscope data
            plt.figure(figsize=(8, 4))
            plt.plot(uncal_df['gyr_x'], label='gyr_x')
            plt.plot(uncal_df['gyr_y'], label='gyr_y')
            plt.plot(uncal_df['gyr_z'], label='gyr_z')
            plt.xlabel('Sampled Time (ms)')
            plt.ylabel('Angular Velocity (deg/s)')
            plt.title('Uncalibrated Gyroscope Data')
            plt.legend()
            plt.show()

            plt.figure(4)
            # Plot gyroscope data
            plt.figure(figsize=(8, 4))
            plt.plot(calibrated_data['gyr_x'], label='gyr_x')
            plt.plot(calibrated_data['gyr_y'], label='gyr_y')
            plt.plot(calibrated_data['gyr_z'], label='gyr_z')
            plt.xlabel('Sampled Time (ms)')
            plt.ylabel('Angular Velocity (deg/s)')
            plt.title('Calibrated Gyroscope Data')
            plt.legend()

            plt.figure(5)
            # Plot gyroscope data
            plt.figure(figsize=(8, 4))
            plt.plot(calibrated_data['gyr_x'], label='gyr_x')
            plt.plot(calibrated_data['gyr_y'], label='gyr_y')
            plt.plot(calibrated_data['gyr_z'], label='gyr_z')
            plt.xlabel('Sampled Time (ms)')
            plt.ylabel('Angular Velocity (deg/s)')
            plt.title('Calibrated Gyroscope Data')
            plt.legend()

            plt.show()

        # We have acc, gyro and timestamp
        elif len(data_obj) == 5 and data_obj[0] in cal_labels:
            calibration_data.append(data_obj)
            capture = False

    # for index, row in calibrated_data.iterrows():
    #     # print(row['acc_x'])
    #     euler_angle, Q_prev, time_prev = euler_values_ahrs(
    #         madgwick_ahrs, Q_prev, time_prev, row)

    #     euler_angles.append(np.array(euler_angle))

    #     if wframe:
    #         for line in wframe:
    #             line.remove()

    #     # Create a rotation object from Euler angles
    #     rot = Rotation.from_euler('xyz', euler_angle)

    #     # Define the axis vectors
    #     x_axis = np.array([1, 0, 0])
    #     y_axis = np.array([0, 1, 0])
    #     z_axis = np.array([0, 0, 1])

    #     # Rotate the axis vectors
    #     x_rotated = rot.apply(x_axis)
    #     y_rotated = rot.apply(y_axis)
    #     z_rotated = rot.apply(z_axis)

    #     # Plot the rotated axis vectors
    #     origin = np.array([0, 0, 0])

    #     wframe = [ax.plot(*zip(origin, x_rotated),
    #                       color='red', linewidth=2)[0]]
    #     wframe.append(ax.plot(*zip(origin, y_rotated),
    #                   color='blue', linewidth=2)[0])
    #     wframe.append(ax.plot(*zip(origin, z_rotated),
    #                   color='green', linewidth=2)[0])

    #     plt.draw()
    #     # plt.pause(.001)

    #     # line = np.array([1, 0, 0])
    #     # zeros_line = np.array([0, 0, 0])
    #     # wframe = [ax.plot(line, zeros_line, zeros_line,
    #     #                   color='red', linewidth=2)[0]]
    #     # wframe.append(ax.plot(line, zeros_line, zeros_line,
    #     #                       color='red', linewidth=2)[0])
    #     # wframe.append(ax.plot(zeros_line, line, zeros_line,
    #     #                       color='blue', linewidth=2)[0])
    #     # wframe.append(ax.plot(zeros_line, zeros_line,
    #     #                       line, color='green', linewidth=2)[0])

    #     plt.draw()
    #     plt.pause(.001)

    # # print(euler_angles)
    # plt.plot(np.array(euler_angles)[:, 0])
    # plt.plot(np.array(euler_angles)[:, 1])
    # plt.plot(np.array(euler_angles)[:, 2])
    # plt.show()


if __name__ == "__main__":
    main_loop()
    # write_loop()


# plot_euler(euler_angle, 45000, 50000, 1000)
