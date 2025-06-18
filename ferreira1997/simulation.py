# run_single_controller.py

import numpy as np
import matplotlib.pyplot as plt

from model.inverted_pendulum import InvePendulum
from controllers import LQR, SlidingMode

# --- Configuration ---
# Change this variable to test different controllers: 'LQR', 'SM', or 'VF'
CONTROLLER_TO_TEST = 'SM'

SIMULATION_TIME = 5.0  # seconds
DT = 0.002

# We use the same initial condition as the paper's demonstration trajectory in Figure 7
# The initial condition used in the trajectory simulation is alpha=0.5. All other state variables equal zero initially.
INITIAL_STATE = np.array([0.0, 0.0, 0.4, 0.0])

if __name__ == '__main__':
    # --- Initialize Environment and All Controllers ---
    env = InvePendulum(dt=DT)

    # The controllers are defined as per the paper
    all_controllers = {
        "LQR": LQR(10.0, 12.60, 48.33,
                   9.09),  # Standard LQR with gains from the paper 
        "SM":
        SlidingMode(env),  # Sliding Mode controller defined in the paper 
        "VF": LQR(0, 30.92, 87.63,
                  20.40)  # Velocity Feedback LQR with gains from the paper 
    }

    # --- Select the Controller to Test ---
    if CONTROLLER_TO_TEST not in all_controllers:
        print(
            f"Error: Controller '{CONTROLLER_TO_TEST}' not found. Please choose from {list(all_controllers.keys())}."
        )
        exit()

    chosen_controller = all_controllers[CONTROLLER_TO_TEST]
    print(f"Simulating with the '{CONTROLLER_TO_TEST}' controller...")

    # --- Run Simulation Loop ---
    state = env.reset(initial_state=INITIAL_STATE)

    history = {'time': [], 'states': [], 'control_effort': []}

    num_steps = int(SIMULATION_TIME / DT)
    for i in range(num_steps):
        # Log data before stepping
        history['time'].append(i * DT)
        history['states'].append(state)

        # Calculate the control action (force u)
        action = chosen_controller.update_control(state)
        history['control_effort'].append(action)

        # Step the simulation
        state = env.step_sim(action)

        # Optional: Stop if the pendulum falls
        # if abs(state[2]) > np.pi / 2:
        #     print("Pendulum has fallen over. Stopping simulation.")
        #     # Fill the rest of the history to keep array lengths consistent
        #     remaining_steps = num_steps - (i + 1)
        #     history['time'].extend([(i + 1 + j) * DT
        #                             for j in range(remaining_steps)])
        #     history['states'].extend([state] * remaining_steps)
        #     history['control_effort'].extend([action] * remaining_steps)
        #     break

    print("Simulation finished.")

    # --- Plot Results ---
    states_history = np.array(history['states'])

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f"Performance of Individual '{CONTROLLER_TO_TEST}' Controller",
        fontsize=16)

    # Plot 1: Position States
    axs[0].plot(history['time'],
                states_history[:, 2],
                label=r'$\alpha(t)$ - Pendulum Angle')
    axs[0].plot(history['time'],
                states_history[:, 0],
                label=r'$x(t)$ - Cart Position',
                linestyle='--')
    axs[0].set_ylabel("Position (m) / Angle (rad)")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title("State Trajectories")

    # Plot 2: Control Effort
    axs[1].plot(history['time'],
                history['control_effort'],
                label=r'$u(t)$ - Force',
                color='g')
    axs[1].set_ylabel("Control Effort (N)")
    axs[1].set_xlabel("Time (s)")
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title("Control Effort Over Time")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
