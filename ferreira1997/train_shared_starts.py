# train_shared_starts.py

import os
import torch
import numpy as np
import random
from tqdm import tqdm

from model.inverted_pendulum import InvePendulum
from controllers import LQR, SlidingMode
from neural_network import PerformanceIndexNN, calculate_cost_U

# --- Configuration ---
NUM_INITIAL_STATES = 5000 # Renamed for clarity based on the new approach
TRAJECTORY_LENGTH = 500
NUM_EPOCHS = 50 
DELTA = 0.5
MODEL_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def generate_shared_start_data(controllers, env, num_initial_states, trajectory_length):
    """
    Generates data by creating a set of random initial states and running
    one trajectory for each controller from each of those shared states.
    """
    print(f"Generating data from {num_initial_states} shared initial states...")
    
    # Initialize a dictionary to hold the transitions for each controller
    datasets = {name: [] for name in controllers.keys()}
    
    for _ in tqdm(range(num_initial_states)):
        # 1. Generate one shared random initial state
        initial_state = np.array([
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
            random.uniform(-0.5, 0.5),
            random.uniform(-1.0, 1.0)
        ])
        
        # 2. For this single starting state, run a trajectory for each controller
        for name, controller_obj in controllers.items():
            state = env.reset(initial_state=initial_state)
            
            for _ in range(trajectory_length):
                action = controller_obj.update_control(state)
                next_state = env.step_sim(action)
                
                # Append the transition to the correct controller's dataset
                datasets[name].append((state, next_state))
                
                state = next_state
                if abs(state[2]) > np.pi / 2:
                    break
                    
    return datasets

if __name__ == '__main__':
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # --- Initialize Environment and Controllers ---
    env = InvePendulum()
    controllers = {
        "LQR": LQR(10.0, 12.60, 48.33, 9.09),
        "SM": SlidingMode(env),
        "VF": LQR(0, 30.92, 87.63, 20.40)
    }

    # The cost matrix P is defined as per the paper
    P = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    p_matrix_device = P.to(DEVICE)

    # --- Generate all data first using the new method ---
    all_datasets = generate_shared_start_data(controllers, env, NUM_INITIAL_STATES, TRAJECTORY_LENGTH)

    # --- Main Training Loop for Each Controller ---
    for name, controller_obj in controllers.items():
        print(f"\n{'='*20}\nTraining model for {name} controller\n{'='*20}")

        # 1. Get the pre-generated data for the current controller
        data = all_datasets[name]
        x_k_full = torch.tensor([item[0] for item in data], dtype=torch.float32).to(DEVICE)
        x_k_plus_1_full = torch.tensor([item[1] for item in data], dtype=torch.float32).to(DEVICE)

        # 2. Setup Model and L-BFGS Optimizer
        model = PerformanceIndexNN().to(DEVICE)
        optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01, max_iter=20)
        loss_function = torch.nn.MSELoss()

        # 3. Training Loop with L-BFGS
        print(f"Starting training for {name} model on {DEVICE} with L-BFGS...")
        model.train()
        for epoch in range(NUM_EPOCHS):
            
            def closure():
                optimizer.zero_grad()
                with torch.no_grad():
                    J_pred_next = model(x_k_plus_1_full)
                    U_k = calculate_cost_U(x_k_full, p_matrix_device)
                    J_target = U_k + DELTA * J_pred_next
                J_pred = model(x_k_full)
                loss = loss_function(J_pred, J_target)
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.6f}")

        # 4. Save the trained model
        model_path = os.path.join(MODEL_DIR, f"{name.lower()}_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model for {name} saved to {model_path}")