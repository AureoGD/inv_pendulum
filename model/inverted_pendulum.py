from math import cos, sin
import numpy as np


class InvePendulum():
    """
    A class to simulate the inverted pendulum system as described in the paper
    "Controller Scheduling by Neural Networks" by Ferreira and Krogh (1997).

    The simulation is based on solving the differential equations of motion
    for the pendulum and cart system.

    Attributes:
        dt (float): The time step for the simulation in seconds.
        Jt (float): Inertia of the cart (base).
        m (float): Mass of the pendulum bar.
        l (float): Length of the pendulum bar.
        br (float): Friction coefficient for the cart.
        g (float): Acceleration due to gravity.
        f_max (float): Maximum applicable force.
        v_max (float): Maximum cart velocity.
        x_max (float): Maximum cart position from the center.
        x (float): Current position of the cart.
        dx (float): Current velocity of the cart.
        a (float): Current angle of the pendulum in radians.
        da (float): Current angular velocity of the pendulum.
    """

    def __init__(self, dt=0.002):
        """
        Initializes the inverted pendulum simulation environment.

        Args:
            dt (float, optional): The simulation time step. Defaults to 0.02.
        """
        # Parameters from Table in "Controller Scheduling by Neural Networks"
        self.Jt = 0.6650  # Inertia of the base (Kg)
        self.m = 0.21  # Mass of the pendulum (Kg)
        self.l = 0.61  # Length of the pendulum (m)
        self.br = 0.2  # Friction coefficient of the base (Kg/s)
        self.g = 9.8  # Gravitational acceleration (m/s^2)

        # System constraints
        self.f_max = 20.0  # Max force (N)
        self.v_max = 3.0  # Max velocity (m/s)
        self.x_max = 2.0  # Max position (m)

        # Simulation time step
        self.dt = dt

        # Initialize state variables
        self.reset()

    def step_sim(self, force):
        """
        Advances the simulation by one time step.

        This method takes an input force, calculates the resulting linear and
        angular accelerations by solving the system's equations of motion,
        and then updates the state variables using Euler integration.

        The equations of motion are:
        Jt*ddx + 0.5*m*l*cos(a)*dda + Br*dx - 0.5*m*l*sin(a)*da^2 = F
        0.5*m*cos(a)*ddx + (1/3)*m*l*dda - 0.5*m*g*sin(a) = 0

        Args:
            force (float): The force applied to the cart.

        Returns:
            tuple: A tuple containing the new state (x, dx, a, da).
        """
        # Clamp the input force to its maximum value
        force = np.clip(force, -self.f_max, self.f_max)

        # Get current state for easier access
        x, dx, a, da = self.x, self.dx, self.a, self.da

        # --- Solve the system of linear equations for accelerations ---
        # The equations can be written in matrix form A * acc = B, where
        # acc is the vector [ddx, dda]^T.

        # Define the A matrix
        a_11 = self.Jt
        a_12 = 0.5 * self.m * self.l * cos(a)
        a_21 = 0.5 * self.m * cos(a)
        a_22 = (1.0 / 3.0) * self.m * self.l
        A = np.array([[a_11, a_12], [a_21, a_22]])

        # Define the B vector
        b_1 = force - self.br * dx + 0.5 * self.m * self.l * (da**2) * sin(a)
        b_2 = 0.5 * self.m * self.g * sin(a)
        B = np.array([b_1, b_2])

        try:
            # Solve for the acceleration vector [ddx, dda]
            acc = np.linalg.solve(A, B)
            ddx, dda = acc[0], acc[1]
        except np.linalg.LinAlgError:
            # If the matrix is singular (e.g., at a=pi/2), accelerations are zero
            ddx, dda = 0.0, 0.0

        # --- Update state using Euler integration ---
        self.dx += ddx * self.dt
        self.x += self.dx * self.dt
        self.da += dda * self.dt
        self.a += self.da * self.dt

        # Enforce state constraints
        self.x = np.clip(self.x, -self.x_max, self.x_max)
        self.dx = np.clip(self.dx, -self.v_max, self.v_max)

        # If the cart hits a boundary, its velocity becomes zero
        if self.x == -self.x_max or self.x == self.x_max:
            self.dx = 0.0

        return self.get_state()

    def reset(self, initial_state=(0.0, 0.0, 0.0, 0.0)):
        """
        Resets the pendulum's state to a specified initial condition.

        Args:
            initial_state (tuple, optional): The initial state (x, dx, a, da).
                                             Defaults to (0.0, 0.0, 0.0, 0.0).

        Returns:
            tuple: The initial state.
        """
        self.x, self.dx, self.a, self.da = initial_state
        return self.get_state()

    def get_state(self):
        """
        Returns the current state of the system.

        Returns:
            tuple: A tuple containing the current state (x, dx, a, da).
        """
        return (self.x, self.dx, self.a, self.da)


# --- Example of how to use the class ---
if __name__ == '__main__':
    # Create an instance of the pendulum simulation
    pendulum = InvePendulum(dt=0.01)

    # Reset the pendulum to a starting angle of 0.5 radians (about 28.6 degrees)
    # This is the initial condition used in the paper's trajectory simulation
    initial_a = 0.5
    state = pendulum.reset(initial_state=(0.0, 0.0, initial_a, 0.0))
    print(
        f"Starting simulation from state: (x={state[0]:.2f}, dx={state[1]:.2f}, a={state[2]:.2f}, da={state[3]:.2f})"
    )

    # Run the simulation for a few steps with a constant force
    applied_force = 1.5  # Newtons
    simulation_time = 2.0  # Seconds
    num_steps = int(simulation_time / pendulum.dt)

    print(
        f"\nApplying a constant force of {applied_force} N for {simulation_time} seconds..."
    )
    for step in range(num_steps):
        state = pendulum.step_sim(applied_force)

        # Print the state every 0.5 seconds
        if (step * pendulum.dt) % 0.5 == 0:
            print(f"Time: {step * pendulum.dt:.2f}s | "
                  f"x: {state[0]:.3f}m | "
                  f"dx: {state[1]:.3f}m/s | "
                  f"Angle: {state[2]:.3f}rad | "
                  f"da: {state[3]:.3f}rad/s")

    print("\nSimulation finished.")
