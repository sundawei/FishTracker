import numpy as np
import math
import os
import cv2
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, t):
        # motion model for a moving object with uniformly accelerated speed along straight line
        # motion model :
        #             X(n) = F * X(n-1) + B * u(n)     -- priori
        #            P(n) = F * P(n-1) * F.T + Q       -- priori
        # measurement / observation model:
        #           Z(n) = H * X(n)
        # measurement update equation:
        #           X(n) = X(n) + K(n) * [Z(n)-H*X(n)]  -- posteriori
        #           P(n) = P(n) - K(n) * H * P(n)       -- posteriori
        #           K = P(n) * H.T / [H*P(n)* H.T + R]
        #       where,
        #            state variable vector X: X = [x, y, theta, vx, vy, w].T
        #            measurement vector Z: Z = [x, y, theta].T
        #            state transition matrix F
        #            input control matrix B
        #            measurement matrix H
        #            state varaible covariance matrix P
        #            measurement noise covariance matrix R
        #            process noise / input control covariance matrix Q
        #            Kalman Gain K

        self.t = t  # time interval
        self.X = np.zeros((6, 1))  # initial state vector
        self.F = np.eye(self.X.shape[0])
        self.F[[0, 1, 2], [3, 4, 5]] = self.t
        self.B = np.vstack((np.eye(3) * self.t ** 2 * 0.5, np.eye(3) * self.t))
        self.H = np.hstack((np.eye(3), np.zeros((3, 3))))

        self.u = np.zeros((3, 1))  # initial input control vector
        self.P = np.eye(6)  # initial state variable covariable matrix
        self.Q = np.eye(6) * 4**2  # initial process noise covaraince matrix
        self.R = np.eye(3) * 2**2  # measurement noise (e.g.,5 pixel) covaraince matrix
        self.K = None

        self.history_u = []  # store history input control vector u

    def init_state_vector(self, x, y, theta):
        # initilize X using the position of the first tracked object of a trace
        self.X = np.array([[x], [y], [theta], [0], [0], [0]]) + \
                 np.random.normal(loc=0, scale=1, size=(6, 1))

    def set_input_control_vector(self, u):
        # begin to update after 3 frames having being tracked
        self.u = np.asarray(u).reshape(3, 1)
        self.history_u.append(self.u)

        if len(self.history_u) > 2:
            if len(self.history_u) > 10:
                self.history_u = self.history_u[-10:]
            # compute input control covariance matrix Qc
            Qc = np.cov(np.concatenate(self.history_u, axis=1))
            # process noise covariance matrix Q = B * Qc * B'
            self.Q = self.B.dot(Qc).dot(self.B.T)

    def adjust_speed_state(self, decay_factor):
        """  decay speed state [vx, vy, w] when special cases occurred.
             speed variables take the last three elements of the state vector
        :param decay_factor: a scalar or an array of size [3, 1]
        """
        self.X[3:] = self.X[3:] * decay_factor

    def predict(self, u=None):
        # if u is not None:
        #     self.set_input_control_vector(u)
        # else:
        #     self.set_input_control_vector(np.zeros((3, 1)))
        self.X = self.F.dot(self.X) + self.B.dot(self.u)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

    def correct(self, Z):
        self.K = np.dot(self.P.dot(self.H.T),
                        np.linalg.inv(self.H.dot(self.P).dot(self.H.T) + self.R))

        self.X = self.X + self.K.dot((Z - self.H.dot(self.X)))
        self.P = (np.eye(self.X.shape[0]) - np.dot(self.K, self.H)).dot(self.P)

    def get_estimate_position(self):
        return np.dot(self.H, self.X)


