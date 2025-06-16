import numpy as np
from math import sin, cos, pi

class LQR():
    def __init__(self, k1, k2, k3, k4):
        self.K = np.array(([k1, k2, k3, k4]))
    
    def update_control(self, states):
        u = self.K@states
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
        self.gamma = 20.0
        
        # Physical parameters of the system to be controlled
        self.Jt = pendulum_params.Jt
        self.m = pendulum_params.m
        self.l = pendulum_params.l
        self.br = pendulum_params.br
        self.g = pendulum_params.g

    def update_control(self, states):
        """
        Calculates the control force based on the current system state.

        Args:
            states (tuple or np.array): The current state vector (x, dx, a, da).

        Returns:
            float: The calculated control force.
        """
        _x, dx, a, da = states
        
        # 1. Calculate the sliding surface value, s(t) 
        s = da + self.lambda_ * a

        # --- Calculate the nominal control force (u_hat) ---
        # This is the force required to keep the system on the surface (s_dot = 0).
        
        # Desired angular acceleration to stay on the surface
        dda_desired = -self.lambda_ * da
        
        # Required linear acceleration, derived from the 2nd eq. of motion
        cos_a = cos(a)
        if abs(cos_a) < 1e-6: # Avoid division by zero if pendulum is horizontal
            return 0.0 
            
        numerator = self.m * self.g * sin(a) + (2/3) * self.m * self.l * self.lambda_ * da
        ddx_required = numerator / (self.m * cos_a)
        
        # Calculate nominal force (u_hat) using the 1st eq. of motion 
        u_hat = (self.Jt * ddx_required +
                 0.5 * self.m * self.l * cos_a * dda_desired +
                 self.br * dx -
                 0.5 * self.m * self.l * sin(a) * da**2)

        # --- Calculate the final control force, u ---
        # Add the term that pushes the system towards the surface 
        u = u_hat + self.gamma * np.clip(s / self.phi, -1, 1)

        return u
