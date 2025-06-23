import numpy as np
import matplotlib.pyplot as plt
from math import pi

from model.inverted_pendulum import InvePendulum
from ferreira1997.controllers import LQR
from swup import SwingUp

# --- Configuration ---
SIMULATION_TIME = 20.0  # seconds
DT = 0.002  # Small dt for accuracy

# Start hanging down
INITIAL_STATE = np.array([-1.0, 0.0, 3, 0.0])  # exactly downward

# Region where stabilizer takes over
STABILIZATION_THRESHOLD = 0.5  # radians

if __name__ == '__main__':
    # --- Initialize ---
    env = InvePendulum(dt=DT)

    swingup_controller = SwingUp(env)
    stabilization_controller = LQR(10.0, 12.60, 48.33, 9.09)

    state = env.reset(initial_state=INITIAL_STATE)
    print(f"Starting simulation (alpha={state[2]:.2f} rad)...")

    history = {'time': [], 'states': [], 'control_effort': [], 'active_controller': []}

    num_steps = int(SIMULATION_TIME / DT)
    count = 0
    for i in range(num_steps):
        # Use unwrapped angle for switching
        angle = (state[2] + np.pi) % (2 * np.pi) - np.pi
        da = state[3]

        if abs(angle) > STABILIZATION_THRESHOLD:
            mode = 0  # SwingUp
            force = swingup_controller.update_control(state)
        else:
            mode = 1  # Stabilize
            force = stabilization_controller.update_control(state)

        force = np.clip(force, -3, 3)

        history['time'].append(i * DT)
        history['states'].append(state)
        history['control_effort'].append(force)
        history['active_controller'].append(mode)

        if (i * DT >= 8) and (i * DT < 8.1):
            force = -5

        state = env.step_sim(force)

    print("Simulation finished.")

    # --- Plot ---
    states_history = np.array(history['states'])
    angles_unwrapped = states_history[:, 2]
    angles_wrapped = (angles_unwrapped + pi) % (2 * pi) - pi

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Swing-Up and Stabilization Control", fontsize=16)

    axs[0].plot(history['time'], angles_wrapped)
    axs[0].axhline(y=STABILIZATION_THRESHOLD, color='r', linestyle='--', label='Stabilization Region')
    axs[0].axhline(y=-STABILIZATION_THRESHOLD, color='r', linestyle='--')
    axs[0].set_ylabel("Angle (rad)")
    axs[0].set_title("Pendulum Angle Trajectory")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(history['time'], states_history[:, 0], 'g-')
    axs[1].set_ylabel("Position (m)")
    axs[1].set_title("Cart Position")
    axs[1].grid(True)

    axs[2].plot(history['time'], history['control_effort'], 'k-')
    axs[2].set_ylabel("Force (N)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_title("Control Effort")
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # --- Animation ---
    print("Rendering animation at 30 FPS...")

    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle

    x_full = states_history[:, 0]
    a_full_unwrapped = angles_unwrapped  # use raw, unwrapped

    FPS = 30
    ANIM_DT = 1 / FPS
    SIM_DT = DT

    stride = int(ANIM_DT / SIM_DT)

    x_anim = x_full[::stride]
    a_anim = a_full_unwrapped[::stride]

    l = env.l

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-l - 0.2, l + 0.2)
    ax2.set_aspect('equal')
    ax2.set_title("Inverted Pendulum Swing-Up Animation (30 FPS)")

    ax2.axhline(0, color='k', linewidth=2)

    cart_width = 0.3
    cart_height = 0.2
    cart = Rectangle((0, 0), cart_width, cart_height, fc='blue', ec='black')
    ax2.add_patch(cart)

    line, = ax2.plot([], [], lw=4, color='red')

    def init():
        cart.set_xy((-cart_width / 2, -cart_height / 2))
        line.set_data([], [])
        return cart, line

    def update(frame):
        cart_x = x_anim[frame]
        cart.set_xy((cart_x - cart_width / 2, -cart_height))  # cart below pivot
        pivot_y = 0
        px = cart_x + l * np.sin(a_anim[frame])
        py = pivot_y + l * np.cos(a_anim[frame])  # <-- FLIP SIGN HERE

        line.set_data([cart_x, px], [pivot_y, py])
        return cart, line

    ani = animation.FuncAnimation(fig2, update, frames=len(x_anim), init_func=init, blit=True, interval=1000 / FPS)

    plt.show()
