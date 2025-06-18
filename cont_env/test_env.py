import numpy as np
import matplotlib.pyplot as plt
from cont_env.pendulum_env import InvPendulumEnv


def run_test():
    # Create the environment
    env = InvPendulumEnv(max_step=5000, rendering=True)

    # Reset environment
    obs, info = env.reset()
    done = False
    total_reward = 0

    print("Episode started")
    history = {'states': [], 'time': [], 'control_effort': []}
    ep = 1

    while not done:
        action = env.action_space.sample()  # Random action for now
        obs, reward, terminated, truncated, info = env.step(action)
        history['states'].append(obs.copy())
        history['control_effort'].append(action)
        history['time'].append(ep * 0.002)
        total_reward += reward
        done = terminated or truncated
        ep += 1

    print("Episode finished.")
    print("Total reward:", total_reward)

    env.close()

    states_history = np.array(history['states'])
    control_effort_history = np.array(history['control_effort'])
    time_history = np.array(history['time'])

    angles_unwrapped = states_history[:, 2] * np.pi
    angles_wrapped = (angles_unwrapped + np.pi) % (2 * np.pi) - np.pi  # Wrap for display

    plot_fig, plot_axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plot_fig.suptitle("Simulation History: Inverted Pendulum Control", fontsize=16)
    plot_axs[0].plot(time_history, angles_wrapped)
    plot_axs[0].set_ylabel("Angle (rad)")
    plot_axs[0].set_title("Pendulum Angle Trajectory")
    plot_axs[0].grid(True)
    # plot_axs[0].legend()
    plot_axs[1].plot(time_history, states_history[:, 0] * env.inv_pendulum.x_max, 'g-')
    plot_axs[1].set_ylabel("Position (m)")
    plot_axs[1].set_title("Cart Position")
    plot_axs[1].grid(True)
    plot_axs[2].plot(time_history, control_effort_history * env.inv_pendulum.f_max, 'k-')
    plot_axs[2].set_ylabel("Force (N)")
    plot_axs[2].set_xlabel("Time (s)")
    plot_axs[2].set_title("Control Effort")
    plot_axs[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show(block=True)  # Show non-blocking


if __name__ == "__main__":
    run_test()
