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
                                      parameters of the pendulum (mc, m, l, br, g).
        """
        # Control parameters from the paper
        self.lambda_ = 5.0  # using lambda_ to avoid keyword conflict
        self.phi = 0.1
        self.gamma = 20.0

        # Physical parameters of the system to be controlled
        self.mc = pendulum_params.mc
        self.mr = pendulum_params.mr
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
            numerator = 0.5 * self.mr * self.g * sin(a) - (1 / 3) * self.mr * self.l * dda_desired
            ddx_required = numerator / (self.mr * cos_a * 0.5)

        u_hat = (self.mc * ddx_required + 0.5 * self.mr * self.l * cos_a * dda_desired + self.br * dx -
                 0.5 * self.mr * self.l * sin(a) * da**2)

        u = u_hat + self.gamma * np.clip(s / self.phi, -1, 1)

        return u
