import pybullet as p
import pybullet_data
import numpy as np
import time
from math import pi
import matplotlib.pyplot as plt

from ferreira1997.controllers import LQR, SlidingMode
from model.inverted_pendulum import InvePendulum


class InvertedPendulumPyBullet:
    """
    Updated class with multiple methods for constraining cart position.
    """

    def __init__(self, dt=0.002, render=False):
        self.dt = dt
        self.mode = p.GUI if render else p.DIRECT
        # --- Connect to PyBullet ---
        # Check if a client is already connected to avoid creating a new one
        self.client = p.connect(self.mode) if p.getConnectionInfo()['isConnected'] == 0 else 0

        # --- System Parameters & Constraints ---
        self.g = 9.8
        self.br = 0.2
        self.f_max = 3.0
        self.v_max = 3.0
        self.x_max = 2.0
        self.da_max = 10

        # --- Soft Wall Parameters ---
        self.boundary_stiffness = 5000  # k: Spring stiffness
        self.boundary_damping = 50  # c: Damping

        # --- Setup Simulation Environment ---
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -self.g)
        p.setPhysicsEngineParameter(fixedTimeStep=self.dt)

        self.plane = p.loadURDF("plane.urdf")

        start_pos = [0, 0, 0.7]  # Lifted to allow swing-up
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.pendulum = p.loadURDF("model/inve_pendulum.urdf", start_pos, start_orn)

        self.joint_indices = {
            p.getJointInfo(self.pendulum, i)[1].decode('UTF-8'): i
            for i in range(p.getNumJoints(self.pendulum))
        }

        p.changeDynamics(self.pendulum, -1, linearDamping=self.br)
        p.changeDynamics(self.pendulum, self.joint_indices['cart_to_rod'], linearDamping=0, angularDamping=0.1)

        # Disable default motors
        for joint_index in self.joint_indices.values():
            p.setJointMotorControl2(self.pendulum, joint_index, p.VELOCITY_CONTROL, force=0)

        self.reset()

    # METHOD 1: SOFT VIRTUAL WALLS
    def step_sim_soft_walls(self, control_force):
        """
        Advances the simulation, applying a spring-damper force at the boundaries.
        """
        x, dx, _, _ = self.get_state()

        # Calculate boundary force
        boundary_force = 0.0
        if x > self.x_max:
            displacement = x - self.x_max
            boundary_force = -self.boundary_stiffness * displacement - self.boundary_damping * dx
        elif x < -self.x_max:
            displacement = x + self.x_max
            boundary_force = -self.boundary_stiffness * displacement - self.boundary_damping * dx

        # Combine control force and boundary force
        total_force = np.clip(control_force, -self.f_max, self.f_max) + boundary_force

        # Apply total force
        p.setJointMotorControl2(self.pendulum,
                                self.joint_indices['slider_to_cart'],
                                p.TORQUE_CONTROL,
                                force=total_force)
        p.stepSimulation()

        return self._enforce_velocity_limits_and_get_state()

    # METHOD 2: HARD POSITION CLAMPING
    def step_sim_hard_clamp(self, control_force):
        """
        Advances the simulation and then manually clamps position if out of bounds.
        """
        # Step 1: Apply force and step simulation as normal
        force = np.clip(control_force, -self.f_max, self.f_max)
        p.setJointMotorControl2(self.pendulum, self.joint_indices['slider_to_cart'], p.TORQUE_CONTROL, force=force)
        p.stepSimulation()

        # Step 2: Get the state from the physics engine
        x, dx, a, da = self.get_state()

        # Step 3: Check if position is out of bounds and clamp if necessary
        if abs(x) > self.x_max:
            # Clamp the position to the boundary
            x_clamped = np.clip(x, -self.x_max, self.x_max)
            # Reset the joint state with the new position and zero velocity
            p.resetJointState(self.pendulum,
                              self.joint_indices['slider_to_cart'],
                              targetValue=x_clamped,
                              targetVelocity=0.0)
            # Update state variables after clamping
            x, dx = x_clamped, 0.0

        return self._enforce_velocity_limits_and_get_state()

    def _enforce_velocity_limits_and_get_state(self):
        """Helper function to clip velocities and return the final state."""
        x, dx, a, da = self.get_state()

        dx_clipped = np.clip(dx, -self.v_max, self.v_max)
        da_clipped = np.clip(da, -self.da_max, self.da_max)

        # If velocity was clipped, we must reset the state to enforce the hard limit
        if dx_clipped != dx or da_clipped != da:
            p.resetJointState(self.pendulum,
                              self.joint_indices['slider_to_cart'],
                              targetValue=x,
                              targetVelocity=dx_clipped)
            p.resetJointState(self.pendulum,
                              self.joint_indices['cart_to_rod'],
                              targetValue=a,
                              targetVelocity=da_clipped)

        return self.get_state()

    def reset(self, initial_state=(0.0, 0.0, 0.0, 0.0)):
        """Resets the pendulum's state."""
        x, dx, a, da = initial_state
        p.resetJointState(self.pendulum, self.joint_indices['slider_to_cart'], targetValue=x, targetVelocity=dx)
        p.resetJointState(self.pendulum, self.joint_indices['cart_to_rod'], targetValue=a, targetVelocity=da)
        return self.get_state()

    def get_state(self):
        """Returns the current state of the system."""
        x, dx = p.getJointState(self.pendulum, self.joint_indices['slider_to_cart'])[:2]
        a, da = p.getJointState(self.pendulum, self.joint_indices['cart_to_rod'])[:2]
        return (x, dx, a, da)

    def close(self):
        if p.getConnectionInfo()['isConnected'] == 1:
            p.disconnect(self.client)


# =============================================================================
# Main Comparative Simulation
# =============================================================================
def run_comparison():
    # --- Simulation Parameters ---
    simulation_time = 4.0
    dt = 0.002
    num_steps = int(simulation_time / dt)
    constant_force = 2.0

    # --- Data Storage for all states ---
    data = {'soft_walls': {'x': [], 'dx': [], 'a': [], 'da': []}, 'hard_clamp': {'x': [], 'dx': [], 'a': [], 'da': []}}
    time_axis = np.linspace(0, simulation_time, num_steps)

    # --- Simulation 1: Soft Walls ---
    print("Running simulation with Soft Walls...")
    env_soft = InvertedPendulumPyBullet(dt=dt)
    env_soft.reset()
    for i in range(num_steps):
        x, dx, a, da = env_soft.step_sim_soft_walls(constant_force)
        data['soft_walls']['x'].append(x)
        data['soft_walls']['dx'].append(dx)
        data['soft_walls']['a'].append(a)
        data['soft_walls']['da'].append(da)
    env_soft.close()

    # --- Simulation 2: Hard Clamp ---
    print("Running simulation with Hard Clamp...")
    env_hard = InvertedPendulumPyBullet(dt=dt)
    env_hard.reset()
    for i in range(num_steps):
        x, dx, a, da = env_hard.step_sim_hard_clamp(constant_force)
        data['hard_clamp']['x'].append(x)
        data['hard_clamp']['dx'].append(dx)
        data['hard_clamp']['a'].append(a)
        data['hard_clamp']['da'].append(da)
    env_hard.close()

    # --- Plotting Results ---
    print("Plotting results...")
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    fig.suptitle('Comparison of Boundary Handling Methods (Full State)', fontsize=16)

    # Position Plot
    axs[0].plot(time_axis, data['soft_walls']['x'], label='Soft Walls (Spring-Damper)', color='b')
    axs[0].plot(time_axis, data['hard_clamp']['x'], label='Hard Clamp (Manual Reset)', color='r', linestyle='--')
    axs[0].axhline(y=env_hard.x_max, color='k', linestyle=':', label=f'x_max = {env_hard.x_max} m')
    axs[0].set_ylabel('Cart Position (x) [m]')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title('Cart Linear States')

    # Velocity Plot
    axs[1].plot(time_axis, data['soft_walls']['dx'], label='Soft Walls', color='b')
    axs[1].plot(time_axis, data['hard_clamp']['dx'], label='Hard Clamp', color='r', linestyle='--')
    axs[1].set_ylabel('Cart Velocity (dx) [m/s]')
    axs[1].legend()
    axs[1].grid(True)

    # Angle Plot
    axs[2].plot(time_axis, data['soft_walls']['a'], label='Soft Walls', color='b')
    axs[2].plot(time_axis, data['hard_clamp']['a'], label='Hard Clamp', color='r', linestyle='--')
    axs[2].set_ylabel('Pendulum Angle (a) [rad]')
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_title('Pendulum Angular States')

    # Angular Velocity Plot
    axs[3].plot(time_axis, data['soft_walls']['da'], label='Soft Walls', color='b')
    axs[3].plot(time_axis, data['hard_clamp']['da'], label='Hard Clamp', color='r', linestyle='--')
    axs[3].set_xlabel('Time [s]')
    axs[3].set_ylabel('Angular Velocity (da) [rad/s]')
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    pendulum = InvertedPendulumPyBullet(dt=0.002, render=True)

    # Reset to hanging down
    state = pendulum.reset(initial_state=(0.0, 0.0, 0.5, 0.0))
    print(f"Starting simulation from state: {state}")
    env_pen = InvePendulum()
    lqr = LQR(10.0, 12.60, 48.33, 9.09)  # Standard LQR with gains from the paper
    sm = SlidingMode(env_pen)  # Sliding Mode controller defined in the paper
    vf = LQR(0, 30.92, 87.63, 20.40)  # Velocity Feedback LQR with gains from the paper

    # --- CHOOSE YOUR METHOD ---
    # To use a method, call it inside the loop.
    # Comment out the one you are not using.

    for step in range(1500):
        x, dx, angle, da = pendulum.get_state()
        st = np.array([x, dx, angle, da])
        control_force = sm.update_control(st)

        # --- Call the desired step function ---

        # Option 1: Soft Walls
        state = pendulum.step_sim_soft_walls(control_force)

        # Option 2: Hard Clamp (like original script)
        # state = pendulum.step_sim_hard_clamp(control_force)

        if pendulum.mode == p.GUI:
            time.sleep(pendulum.dt)

    print("\nSimulation finished.")
    pendulum.close()
