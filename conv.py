import numpy as np
import matplotlib.pyplot as plt
import math
# import numpy as np
import os

file_path = '/home/swayam/Downloads/groundTruth.txt'

# Initialize arrays to store the parsed values
x_values = []
y_values = []
theta_values = []

# Read the text file line by line
with open(file_path, 'r') as file:
    for line in file:
        # Split the line by whitespace and convert values to floats
        values = line.strip().split()
        x = float(values[0])
        y = float(values[1])
        theta = float(values[2])

        # Append the values to respective arrays
        x_values.append(x)
        y_values.append(y)
        theta_values.append(theta)

def readG2o(fileName):
    f = open(fileName, 'r')
    A = f.readlines()
    f.close()

    X = []
    Y = []
    THETA = []

    for line in A:
        if "VERTEX_SE2" in line:
            line = line.strip()
            (ver, ind, x, y, theta) = line.split(' ')
            X.append(float(x))
            Y.append(float(y))
            THETA.append(float(theta.rstrip('\n')))

    return (X, Y, THETA)

# file_path = '/home/swayam/noise.g2o'

# Initialize arrays to store the parsed values
# xN = []
# yN = []
# tN = []

# Read the text file line by line
# with open(file_path, 'r') as file:
#     for line in file:
#         # Split the line by whitespace and convert values to floats
#         values = line.strip().split()
#         x = float(values[0])
#         y = float(values[1])
#         theta = float(values[2])

#         # Append the values to respective arrays
#         xN.append(x)
#         yN.append(y)
#         tN.append(theta)

(xN,yN,tN) = readG2o("noise.g2o")

(xOpt, yOpt, tOpt) = readG2o("opt.g2o")

def drawThree(X1, Y1, THETA1, X2, Y2, THETA2, X3, Y3, THETA3):
    ax = plt.subplot(111)
    ax.plot(X1, Y1, 'ro', label='Ground Truth')
    plt.plot(X1, Y1, 'k-')

    # for i in range(len(THETA1)):
    #     x2 = 0.25*math.cos(THETA1[i]) + X1[i]
    #     y2 = 0.25*math.sin(THETA1[i]) + Y1[i]
    #     plt.plot([X1[i], x2], [Y1[i], y2], 'r->')

    ax.plot(X2, Y2, 'bo', label='Optimized')
    plt.plot(X2, Y2, 'k-')

    # for i in range(len(THETA2)):
    #     x2 = 0.25*math.cos(THETA2[i]) + X2[i]
    #     y2 = 0.25*math.sin(THETA2[i]) + Y2[i]
    #     plt.plot([X2[i], x2], [Y2[i], y2], 'b->')

    ax.plot(X3, Y3, 'go', label='Noisy')
    plt.plot(X3, Y3, 'k-')

    # for i in range(len(THETA3)):
    #     x2 = 0.25*math.cos(THETA3[i]) + X3[i]
    #     y2 = 0.25*math.sin(THETA3[i]) + Y3[i]
    #     plt.plot([X3[i], x2], [Y3[i], y2], 'g->')

    plt.legend()
    plt.show()

# print("hello")
# # measuring APE/ATE
# cmd = "evo_ape kitti gt.kitti noise.kitti  -va --plot --plot_mode xy"
# os.system(cmd)
# cmd = "evo_ape kitti gt.kitti opt.kitti  -va --plot --plot_mode xy"
# os.system(cmd)
drawThree(x_values, y_values, theta_values, xOpt, yOpt, tOpt, xN, yN, tN)