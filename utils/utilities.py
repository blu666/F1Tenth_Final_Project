import numpy as np
import pdb
import datetime

def Regression(x, u, lamb):
    """Estimates linear system dynamics
    x, u: date used in the regression
    lamb: regularization coefficient
    """

    # Want to solve W^* = argmin sum_i ||W^T z_i - y_i ||_2^2 + lamb ||W||_F,
    # with z_i = [x_i u_i] and W \in R^{n + d} x n
    Y = x[2:x.shape[0], :]
    X = np.hstack((x[1:(x.shape[0] - 1), :], u[1:(x.shape[0] - 1), :]))

    Q = np.linalg.inv(np.dot(X.T, X) + lamb * np.eye(X.shape[1]))
    b = np.dot(X.T, Y)
    W = np.dot(Q, b)

    A = W.T[:, 0:6]
    B = W.T[:, 6:8]

    ErrorMatrix = np.dot(X, W) - Y
    ErrorMax = np.max(ErrorMatrix, axis=0)
    ErrorMin = np.min(ErrorMatrix, axis=0)
    Error = np.vstack((ErrorMax, ErrorMin))

    return A, B, Error


def load_init_ss(path, length=10, track_length=0):
    data = np.loadtxt(path, delimiter=',', usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12)) # (N, 13)
    time, lap, vx, vy, wz, epsi, s, ey, yaw, X, Y, u = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6], data[:, 7], data[:, 8], data[:, 9], data[:, 10], data[:, 11:]
    # print(data.shape)
    # NOTE: xPID_cl is [vx, vy, wz, epsi, s, ey]; xPID_cl_glob is [vx, vy, wz, psi, X, Y]; u is [delta, a]
    
    xPID_cl = np.vstack((vx, vy, wz, epsi, s, ey)).T # (n, 6)
    xPID_cl_glob = np.vstack((vx, vy, wz, yaw, X, Y)).T # (n, 6)
    uPID_cl = u # (n, 2)
    xPID_cls = []
    uPID_cls = []
    xPID_cl_globs = []
    prev_start_idx = 0
    mid_start_idx = 0
    # print(lap[prev_start_idx])
    for i in range(1, xPID_cl.shape[0]):
        if lap[i] - lap[prev_start_idx] == 1 and mid_start_idx == prev_start_idx:
            mid_start_idx = i
        if lap[i] - lap[prev_start_idx] == 2:
            # print(lap[prev_start_idx], lap[i])
            xPID_2laps = xPID_cl[prev_start_idx:i, :]
            xPID_2laps[mid_start_idx:, 4] += track_length
            xPID_cls.append(xPID_2laps)
            uPID_cls.append(uPID_cl[prev_start_idx:i, :])
            xPID_cl_globs.append(xPID_cl_glob[prev_start_idx:i, :])
            prev_start_idx = i
            mid_start_idx = i
    
    xPID_2laps = xPID_cl[prev_start_idx:i, :]
    xPID_2laps[mid_start_idx:, 4] += track_length
    xPID_cls.append(xPID_2laps)
    uPID_cls.append(uPID_cl[prev_start_idx:i, :])
    xPID_cl_globs.append(xPID_cl_glob[prev_start_idx:i, :])
    # if prev_start_idx < xPID_cl.shape[0]:
    #     xPID_cls.append(xPID_cl[prev_start_idx:, :])
    #     uPID_cls.append(uPID_cl[prev_start_idx:, :])
    #     xPID_cl_globs.append(xPID_cl_glob[prev_start_idx:, :])
    xPID_cls = xPID_cls
    uPID_cls = uPID_cls
    xPID_cl_globs = xPID_cl_globs
    # print(len(xPID_cls), xPID_cls[0].shape)
    return xPID_cls, uPID_cls, xPID_cl_globs


def wrap(angle):
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle


class PID:
    """Create the PID controller used for path following at constant speed
    Attributes:
        solve: given x0 computes the control action
    """
    def __init__(self, vt):
        """Initialization
        Arguments:
            vt: target velocity
        """
        self.vt = vt
        self.uPred = np.zeros([1,2])

        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.linearizationTime = deltaTimer
        self.feasible = 1

    def solve(self, x0):
        """Computes control action
        Arguments:
            x0: current state position
        """
        vt = self.vt
        self.uPred[0, 0] = - 0.6 * x0[5] - 0.9 * x0[3] + np.max([-0.9, np.min([np.random.randn() * 0.25, 0.9])])
        self.uPred[0, 1] = 1.5 * (vt - x0[0]) + np.max([-0.2, np.min([np.random.randn() * 0.10, 0.2])])

if __name__ == "__main__":
    x0_cls, u0_cls, x0_cl_globs = load_init_ss('./map/initial_ss.csv', 5)
    print(x0_cls[0][:5])