# run_switching_controller.py (CUDA-enabled version)

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model.inverted_pendulum import InvePendulum
from controllers import LQR, SlidingMode
from neural_network import PerformanceIndexNN

# --- Configuration ---
# --- Configuration ---
MODEL_DIR = "models"
SIMULATION_TIME = 5.0 # seconds
DT = 0.015
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

if __name__ == '__main__':
    # --- Check for trained models ---
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Directory '{MODEL_DIR}' not found. Please run train.py first.")
        exit()

    # --- Initialize Environment and Controllers ---
    env = InvePendulum(dt=DT)
    
    controller_configs = {
        "LQR": LQR(10.0, 12.60, 48.33, 9.09), # Standard LQR gains from the paper 
        "SM": SlidingMode(env),              # Sliding Mode controller 
        "VF": LQR(0, 30.92, 87.63, 20.40)     # Velocity Feedback (VF) LQR gains from the paper 
    }
    controllers = list(controller_configs.values())
    controller_names = list(controller_configs.keys())
    
    # --- Load Trained Neural Network Models ---
    models = []
    print("Loading trained models...")
    for name in controller_names:
        model_path = os.path.join(MODEL_DIR, f"{name.lower()}_model.pth")
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found. Please run train.py first.")
            exit()
            
        # Instantiate the model, move to device, then load the state dictionary
        model = PerformanceIndexNN().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval() # Set model to evaluation mode
        models.append(model)
    print("Models loaded successfully.")

    # --- Run Simulation with Switching Logic ---
    print("Running simulation with switching controller...")
    # The paper uses an initial condition of alpha = 0.5 for its trajectory simulation 
    initial_state = np.array([0.0, 0.0, 0.15, 0.0]) 
    state = env.reset(initial_state=initial_state)

    history = {
        'time': [],
        'states': [],
        'chosen_controller': []
    }

    num_steps = int(SIMULATION_TIME / DT)
    for i in range(num_steps):
        # Move the current state to a tensor on the correct device for model inference
        current_state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Use the neural networks to evaluate the future performance of each controller 
        with torch.no_grad():
            costs = [model(current_state_tensor).item() for model in models]
        
        # Select the controller with the minimum predicted cost
        best_controller_index = np.argmin(costs)
        best_controller = controllers[best_controller_index]
        
        # Apply the chosen controller's action
        action = best_controller.update_control(state)
        state = env.step_sim(action)
        
        # Log data for plotting
        history['time'].append(i * DT)
        history['states'].append(state)
        history['chosen_controller'].append(best_controller_index)

    print("Simulation finished.")

    # --- Plot Results ---
    # This section aims to replicate the plots shown in Figure 8 of the paper 
    states_history = np.array(history['states'])
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Controller Switching Simulation (Replication of Fig. 8)", fontsize=16)

    # Plot 1: Pendulum Angle (alpha) and Cart Position (x)
    axs[0].plot(history['time'], states_history[:, 2], label=r'$\alpha(t)$ - Pendulum Angle')
    axs[0].plot(history['time'], states_history[:, 0], label=r'$x(t)$ - Cart Position', linestyle='--')
    axs[0].set_ylabel("Position (m) / Angle (rad)")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title("System State Trajectories")

    # Plot 2: Active Controller
    axs[1].plot(history['time'], history['chosen_controller'], drawstyle='steps-post', label='Active Controller')
    axs[1].set_yticks(range(len(controller_names)))
    axs[1].set_yticklabels(controller_names)
    axs[1].set_ylabel("Controller")
    axs[1].set_xlabel("Time (s)")
    axs[1].grid(True)
    axs[1].set_title("Controller Selection Over Time")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()