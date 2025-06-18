import numpy as np
from math import sin, cos, pi


class LQR():

    def __init__(self, k1, k2, k3, k4):
        self.K = np.array(([k1, k2, k3, k4]))

    def update_control(self, states):
        u = self.K @ states
        return u


class SlidingMode():
    """
    Implements the Sliding Mode (SM) controller from the paper 
    "Controller Scheduling by Neural Networks" by Ferreira and Krogh (1997).
    """

    def __init__(self, pendulum_params):
        """
        Initializes the SM controller with system and control parameters.

        Args:
            pendulum_params (object): An object or dict containing the physical
                                      parameters of the pendulum (Jt, m, l, br, g).
        """
        # Control parameters from the paper
        self.lambda_ = 5.0  # using lambda_ to avoid keyword conflict
        self.phi = 0.1
        self.gamma = 2.0

        # Physical parameters of the system to be controlled
        self.Jt = pendulum_params.Jt
        self.m = pendulum_params.m
        self.l = pendulum_params.l
        self.br = pendulum_params.br
        self.g = pendulum_params.g

    def update_control(self, states):
        _x, dx, a, da = states

        s = da + self.lambda_ * a

        dda_desired = -self.lambda_ * da
        cos_a = cos(a)

        # Robust: avoid division by near-zero
        if abs(cos_a) < 1e-3:
            ddx_required = 0.0
        else:
            numerator = self.m * self.g * sin(a) - (
                2 / 3) * self.m * self.l * dda_desired
            ddx_required = numerator / (self.m * cos_a)

        u_hat = (self.Jt * ddx_required +
                 0.5 * self.m * self.l * cos_a * dda_desired + self.br * dx -
                 0.5 * self.m * self.l * sin(a) * da**2)

        u = np.clip(u_hat, -1.0,
                    1.0) + self.gamma * np.clip(s / self.phi, -1, 1)

        return u
