import numpy as np
from math import cos, sin


class SwingUp:
    """
    An energy-based swing-up controller.
    It injects energy into the pendulum to swing it up to the top.
    """

    def __init__(self, pendulum_params, gain=1.1):
        """
        Args:
            pendulum_params: An object or namespace with m, l, g, f_max.
            gain (float): Energy control gain (tune as needed).
        """
        self.Jt = pendulum_params.Jt
        self.m = pendulum_params.m
        self.l = pendulum_params.l
        self.g = pendulum_params.g
        self.f_max = pendulum_params.f_max

        # Moment of inertia of a rod pivoting at the base
        self.I = (1 / 3) * self.m * self.l**2

        # Desired total energy at the upright position
        self.E_desired = self.m * self.g * self.l

        # Gain for energy pump
        self.k = gain

    def update_control(self, states):
        """
        Compute swing-up force based on energy difference.
        Args:
            states (tuple): (x, dx, a, da)
        Returns:
            float: Force to apply to the cart.
        """
        _, _, a, da = states

        # Current pendulum energy
        E = 0.5 * self.I * da**2 - self.m * self.g * self.l * cos(a)

        # Energy pumping law: sign synchronizes with swing phase
        u = -self.k * E * np.sign(-da)

        return u
