import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise, Saver


class BasicUKF(UKF):
    def __init__(self, dt=0.1, x_init=np.ones(12)) -> None:
        self.dt = dt

        self.sigma_points = MerweScaledSigmaPoints(n=12, alpha=0.1, beta=2.0, kappa=0.0)
        self.ukf = UKF(dim_x=12, dim_z=4, dt=self.dt, fx=self.f_ca, hx=self.h_ca, points=self.sigma_points) # 12 [x, y, z;
                                                                                                            #     x_dot, y_dot, z_dot;
                                                                                                            #     x_dot_dot, y_dot_dot, z_dot_dot;
                                                                                                            #     yaw, yaw_dot, yaw_dot_dot]
        
        # Set the process noise
        std_f_xy = 1
        std_f_z = 0.3
        std_f_yaw = 0.5

        # Build the process noise matrix
        self.ukf.Q = np.zeros((12, 12))
        self.ukf.Q[0:3, 0:3] = Q_discrete_white_noise(dim=3, dt=self.dt, var=std_f_xy**2)  # Q for x, x_dot, x_dot_dot
        self.ukf.Q[3:6, 3:6] = Q_discrete_white_noise(dim=3, dt=self.dt, var=std_f_xy**2)  # Q for y, y_dot, y_dot_dot
        self.ukf.Q[6:9, 6:9] = Q_discrete_white_noise(dim=3, dt=self.dt, var=std_f_z**2)   # Q for z, z_dot, z_dot_dot
        self.ukf.Q[9:12, 9:12] = Q_discrete_white_noise(dim=3, dt=self.dt, var=std_f_yaw**2) # Q for yaw, yaw_dot, yaw_dot_dot

        # Set the measurement noise
        std_r_xy = 0.02
        std_r_z = 1
        std_r_yaw = 0.1
        self.ukf.R = np.array([[std_r_xy**2, 0, 0, 0],
                               [0, std_r_xy**2, 0, 0],
                               [0, 0, std_r_z**2, 0],
                               [0, 0, 0, std_r_yaw**2]])

        # Set the initial state
        self.ukf.x = x_init # initial state
        self.ukf.P = np.diag([300**2, 3**2, 0.3**2,     # x, x_dot, x_dot_dot
                              300**2, 3**2, 0.3**2,     # y, y_dot, y_dot_dot
                              300**2, 3**2, 0.3**2,     # z, z_dot, z_dot_dot
                              0.3**2, 0.001**2, 0.001**2])    # yaw, yaw_dot, yaw_dot_dot
        
                # Implement residuals to manage angle wrap
        self.ukf.residual_x = self.residual_x
        self.ukf.residual_z = self.residual_z

        
    def f_ca(self, x: np.array, dt: float) -> np.array:

        # Our model is:
        # x[0] = x
        # x[1] = x_dot
        # x[2] = x_dot_dot
        # x[3] = y
        # x[4] = y_dot
        # x[5] = y_dot_dot
        # x[6] = z
        # x[7] = z_dot
        # x[8] = z_dot_dot
        # x[9] = yaw
        # x[10] = yaw_dot
        # x[11] = yaw_dot_dot

        # Initialize the output
        xout = x.copy()

        # Constant acceleration model
        xout[0] += x[1] * dt + 0.5 * x[2] * dt**2
        xout[1] += x[2] * dt
        xout[2] += 0
        xout[3] += x[4] * dt + 0.5 * x[5] * dt**2
        xout[4] += x[5] * dt
        xout[5] += 0
        xout[6] += x[7] * dt + 0.5 * x[8] * dt**2
        xout[7] += x[8] * dt
        xout[8] += 0
        xout[9] += x[10] * dt
        xout[10] += 0
        # xout[9] += x[10] * dt + 0.5 * x[11] * dt**2
        # xout[10] += x[11] * dt
        xout[11] += 0

        return xout
    
    def f_cv(self, x: np.array, dt: float) -> np.array:
        # Our model is:
        # x[0] = x
        # x[1] = x_dot
        # x[2] = x_dot_dot
        # x[3] = y
        # x[4] = y_dot
        # x[5] = y_dot_dot
        # x[6] = z
        # x[7] = z_dot
        # x[8] = z_dot_dot
        # x[9] = yaw
        # x[10] = yaw_dot
        # x[11] = yaw_dot_dot

        # Initialize the output
        xout = x.copy()

        # Constant velocity model
        xout[0] += x[1] * dt
        xout[1] += 0
        xout[2] += 0
        xout[3] += x[4] * dt
        xout[4] += 0
        xout[5] += 0
        xout[6] += x[7] * dt
        xout[7] += 0
        xout[8] += 0
        xout[9] += x[10] * dt
        xout[10] += 0
        xout[11] += 0
        
        return xout
    
    def f_const(self, x: np.array, dt: float) -> np.array:
        return x

    def h_ca(self, x: np.array) -> np.array:
        return np.array([x[0], x[3], x[6], x[9]]) # [x, y, z, yaw]

    def predict_ukf(self, dt=-1.0, **predict_args) -> None:
        if dt == -1.0:
            dt = self.dt
        self.ukf.predict(dt=dt, **predict_args)

    def update_ukf(self, z: np.array, **update_args) -> None:
        self.ukf.update(z, **update_args)

    def residual_x(self, x_pred: np.array, x_actual: np.array) -> np.array:
        """
        Compute the residual (difference) for the state vector x (x_pred - x_actual).
        """
        # Residual for the state vector
        return np.array([x_pred[0] - x_actual[0],
                         x_pred[1] - x_actual[1],
                         x_pred[2] - x_actual[2],
                         x_pred[3] - x_actual[3],
                         x_pred[4] - x_actual[4],
                         x_pred[5] - x_actual[5],
                         x_pred[6] - x_actual[6],
                         x_pred[7] - x_actual[7],
                         x_pred[8] - x_actual[8],
                         self.normalize_angle(self.normalize_angle(x_pred[9]) - 
                                              self.normalize_angle(x_actual[9])),
                         x_pred[10] - x_actual[10],
                         x_pred[11] - x_actual[11]])
    
    def residual_z(self, z_pred: np.array, z_actual: np.array) -> np.array:
        """
        Compute the residual (difference) for the measurement vector z (z_pred - z_actual).
        """

        # Residual for the measurement vector
        return np.array([z_pred[0] - z_actual[0],
                         z_pred[1] - z_actual[1],
                         z_pred[2] - z_actual[2],
                         self.normalize_angle(self.normalize_angle(z_pred[3]) - 
                                              self.normalize_angle(z_actual[3]))])

    def normalize_angle(self, x: float) -> float:
        # Normalize the angle to be in the range [-pi, pi]
        # x = x % (2 * np.pi)
        # if x > np.pi:
        #     x -= 2 * np.pi
        # return x
        # Normalize the angle to be in the range [-pi/2, pi/2]
        while x > np.pi / 2:
            x -= np.pi
        while x < -np.pi / 2:
            x += np.pi
        return x
    
    
    def get_pose(self) -> np.array:
        """
        Get the 3D pose of the tracked object.
        Returns:
            np.array: The 3D position of the object [x, y, z, yaw].
        """
        x = self.ukf.x[0]
        y = self.ukf.x[3]
        z = self.ukf.x[6]
        yaw = self.ukf.x[9]

        return np.array([x, y, z, yaw])


if __name__ == "__main__":
    
    # Test the BasicUKF class

    from numpy.random import randn
    
    # Create an instance of the BasicUKF class
    dt = 0.1
    ukf = BasicUKF(dt=dt, x_init=np.ones(12))

    # Generate some random measurements
    n_measurements = 100
    zs_noNoise = [[i, i, i, i] for i in range(n_measurements)] # measurements [x, y, z, yaw]
    zs_noNoise = (np.array(zs_noNoise) * dt)

    # Add some noise to the measurements
    std_r_xy = 0.20
    std_r_z = 1
    std_r_yaw = 0.1
    zs = zs_noNoise.copy() + np.array([[randn()*std_r_xy, randn()*std_r_xy, randn()*std_r_z, randn()*std_r_yaw] for _ in range(n_measurements)])

    x_states = []

    # Make predictions and updates
    for z in zs:
        ukf.predict_ukf(fx=ukf.f_cv, dt=dt) # process noise
        ukf.update_ukf(z, hx=ukf.h_ca)

        print(ukf.get_pose())

        # # 3 updates before the next prediction
        # ukf.predict_ukf(fx=ukf.f_const, dt=dt)
        # ukf.update_ukf(z, hx=ukf.h_ca)

        # ukf.predict_ukf(fx=ukf.f_const, dt=dt)
        # ukf.update_ukf(z, hx=ukf.h_ca)

        # ukf.predict_ukf(fx=ukf.f_const, dt=dt)
        # ukf.update_ukf(z, hx=ukf.h_ca)

        x_states.append(ukf.ukf.x)
    
    # Save an image of the followed path
    import matplotlib.pyplot as plt
    # Print x, y, z, yaw in the same subfigure
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot([x[0] for x in x_states], [i for i in range(len(x_states))], 'r')
    plt.plot([x[0] for x in zs_noNoise], [i for i in range(len(zs))], alpha=0.5)
    plt.scatter([x[0] for x in zs], [i for i in range(len(zs))], c='r', s=1)
    plt.title('X, X (measurements)')
    plt.legend(['X', 'X (Actual)', 'Z (measurements)'])

    plt.subplot(2, 2, 2)
    plt.plot([x[3] for x in x_states], [i for i in range(len(x_states))], 'g')
    plt.plot([x[1] for x in zs_noNoise], [i for i in range(len(zs))], alpha=0.5)
    plt.scatter([x[1] for x in zs], [i for i in range(len(zs))], c='g', s=1)
    plt.title('Y, Y (measurements)')
    plt.legend(['Y', 'Y (Actual)', 'Z (measurements)'])

    plt.subplot(2, 2, 3)
    plt.plot([x[6] for x in x_states], [i for i in range(len(x_states))], 'b')
    plt.plot([x[2] for x in zs_noNoise], [i for i in range(len(zs))], alpha=0.5)
    plt.scatter([x[2] for x in zs], [i for i in range(len(zs))], c='b', s=1)
    plt.title('Z, Z (measurements)')
    plt.legend(['Z', 'Z (Actual)', 'Z (measurements)'])

    plt.subplot(2, 2, 4)
    # Scatter plot of the measurements
    plt.plot([i for i in range(len(x_states))], [x[9] for x in x_states], 'y')
    plt.plot([i for i in range(len(zs))], [x[3] for x in zs_noNoise], alpha=0.5)
    plt.scatter([i for i in range(len(zs))], [x[3] for x in zs], c='y', s=1)
    plt.title('Yaw, Yaw (measurements)')
    plt.legend(['Yaw', 'Yaw (Actual)', 'Z (measurements)'])
    
    plt.tight_layout()
    plt.savefig('/data/output/figures/ukf_tracking.pdf', dpi=300)

    # Save a plot of the actual path
    plt.figure(figsize=(10, 10))
    plt.plot([x[0] for x in x_states], [x[3] for x in x_states], 'r')
    plt.plot([x[0] for x in zs_noNoise], [x[1] for x in zs_noNoise], alpha=0.5)
    plt.scatter([x[0] for x in zs], [x[1] for x in zs], c='r', s=1)
    plt.title('Actual path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(['X', 'X (Actual)', 'Z (measurements)'])
    plt.tight_layout()
    # save as pdf
    plt.savefig('/data/output/figures/ukf_tracking_path.pdf', dpi=300)

    print(ukf.ukf.P)
