import torch
import numpy as np
import os
import time
from model.inverted_pendulum import InvePendulum
from controllers import LQR, SlidingMode
from neural_network import PerformanceIndexNN
from model.pendulum_animation import PendulumLiveRenderer
from ferreira1997.switching import AdvancedSwitcher
from ferreira1997.plot_sim import SimulationPlotter

# --- Configuration ---
MODEL_DIR = "nn_models/cbnn"
SIMULATION_TIME = 5.0
DT = 0.002
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def run_simulation():
    """
    Main function to load models, run the simulation with the advanced switching
    controller, and return the results.
    """
    # --- 1. Initialize Environment and Controllers ---
    env = InvePendulum(dt=DT)
    controller_configs = {"LQR": LQR(10.0, 12.60, 48.33, 9.09), "SM": SlidingMode(env), "VF": LQR(0, 30.92, 87.63, 20.40)}
    controller_names = list(controller_configs.keys())

    # --- 2. Load Trained Neural Network Models ---
    models = {}
    print("Loading trained models...")
    for name in controller_names:
        model_path = os.path.join(MODEL_DIR, f"{name.lower()}_model.pth")
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found. Please run train.py first.")
            return None, None

        model = PerformanceIndexNN().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        models[name] = model
    print("All models loaded successfully.")

    # --- 3. Run Simulation with Advanced Switching Logic ---
    print("\nRunning real-time simulation with advanced switching controller...")
    initial_state = np.array([0.0, 0.0, 0.4, 0.0])
    state = env.reset(initial_state=initial_state)
    pendulum_renderer = PendulumLiveRenderer(env)
    pendulum_renderer.init_live_render()

    history = {
        'time': [],
        'x': [],
        'angle': [],
        'dx': [],
        'da': [],
        'active_controller': [],
        'j_lqr': [],
        'j_sm': [],
        'j_vf': [],
        'j_true': [],
        'action': []
    }
    num_steps = int(SIMULATION_TIME / DT)

    # Initialize the advanced switcher
    switcher = AdvancedSwitcher(controller_names)

    for i in range(num_steps):
        current_state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            j_predictions = {name: model(current_state_tensor).item() for name, model in models.items()}

        # Use the switcher to select the controller
        best_controller_name = switcher.select_controller(j_predictions)
        best_controller = controller_configs[best_controller_name]

        # Update the switcher's internal state for the next step
        switcher.update_state(best_controller_name, j_predictions)

        action = best_controller.update_control(state)
        state = env.step_sim(action)

        # Log data
        history['time'].append(i * DT)
        history['x'].append(state[0])
        history['dx'].append(state[1])
        history['angle'].append(state[2])
        history['da'].append(state[3])
        history['active_controller'].append(controller_names.index(best_controller_name))
        history['j_lqr'].append(j_predictions['LQR'])
        history['j_sm'].append(j_predictions['SM'])
        history['j_vf'].append(j_predictions['VF'])
        history['action'].append(action)
        if (i % 30 == 0):
            pendulum_renderer.update_live_render()
        time.sleep(env.dt)

    print("\nSimulation finished.")
    pendulum_renderer.close_render()
    return history, controller_names


def plot_simulation_results(history, controller_names):
    """
    Initializes and uses the SimulationPlotter class to visualize results.
    """
    print("Generating plots using SimulationPlotter...")
    # Create an instance of the reusable plotter class
    plotter = SimulationPlotter(controller_names)
    # Call its plot method
    plotter.plot(history, title="Controller Switching Test Results")


if __name__ == '__main__':
    if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
        print(f"Error: Directory '{MODEL_DIR}' is empty or does not exist. Please run train.py first.")
    else:
        simulation_history, names = run_simulation()
        if simulation_history:
            plot_simulation_results(simulation_history, names)
