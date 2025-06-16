# train.py

import os
import torch
import numpy as np
import random
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader
from model.inverted_pendulum import InvePendulum
from controllers import LQR, SlidingMode
from neural_network import PerformanceIndexNN, calculate_cost_U

# --- Configuration ---
# The paper mentions using 5000 random trajectories for off-line training 
NUM_TRAJECTORIES = 5000
TRAJECTORY_LENGTH = 150 # Max steps per trajectory (150 steps * 0.02s/step = 3s)
NUM_EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
# The discount factor delta is specified in the paper 
DELTA = 0.5
MODEL_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def generate_data_for_controller(controller, env, num_trajectories, trajectory_length):
    """Generates state transition data (x_k, x_{k+1}) for a given controller."""
    print(f"Generating {num_trajectories} trajectories...")
    transitions = []
    for _ in tqdm(range(num_trajectories)):
        # Start from a random state to cover the state space
        initial_state = np.array([
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
            random.uniform(-0.5, 0.5), # Angle in radians
            random.uniform(-1.0, 1.0)
        ])
        state = env.reset(initial_state=initial_state)

        for _ in range(trajectory_length):
            action = controller.update_control(state)
            next_state = env.step_sim(action)
            
            transitions.append((state, next_state))
            state = next_state
            
            # Stop if pendulum falls over
            if abs(state[2]) > np.pi / 2:
                break
    return transitions

if __name__ == '__main__':
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # --- Initialize Environment and Controllers ---
    env = InvePendulum()
    
    # The controllers are defined as per the paper
    controllers = {
        "LQR": LQR(10.0, 12.60, 48.33, 9.09),          # Standard LQR with gains from the paper 
        "SM": SlidingMode(env),                         # Sliding Mode controller defined in the paper 
        "VF": LQR(0, 30.92, 87.63, 20.40)              # Velocity Feedback LQR with gains from the paper 
    }

    # The cost matrix P is explicitly defined in the paper 
    P = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    p_matrix_device = P.to(DEVICE)

    # --- Main Training Loop for Each Controller ---
    # A separate neural network is trained to evaluate the performance of each controller 
    for name, controller_obj in controllers.items():
        print(f"\n{'='*20}\nTraining model for {name} controller\n{'='*20}")

        # 1. Generate Data
        data = generate_data_for_controller(controller_obj, env, NUM_TRAJECTORIES, TRAJECTORY_LENGTH)
        
        # Prepare data for PyTorch
        x_k_data = torch.tensor([item[0] for item in data], dtype=torch.float32)
        x_k_plus_1_data = torch.tensor([item[1] for item in data], dtype=torch.float32)
        
        dataset = TensorDataset(x_k_data, x_k_plus_1_data)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 2. Setup Model and Optimizer
        model = PerformanceIndexNN().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_function = torch.nn.MSELoss()

        # 3. Training Loop using Heuristic Dynamic Programming (HDP) 
        print(f"Starting training for {name} model on {DEVICE}...")
        model.train()
        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            for x_k_batch, x_k_plus_1_batch in dataloader:
                # Move data batches to the selected device
                x_k_batch = x_k_batch.to(DEVICE)
                x_k_plus_1_batch = x_k_plus_1_batch.to(DEVICE)
                
                # The target calculation is based on the HDP update rule from the paper (Eq. 4) 
                with torch.no_grad():
                    # Get the network's cost prediction for the *next* state
                    J_pred_next = model(x_k_plus_1_batch)
                    
                    # Calculate the immediate cost U(x) = x^T*P*x for the current state 
                    U_k = calculate_cost_U(x_k_batch, p_matrix_device)

                    # Calculate the target J value: J*(x_k) = U(x_k) + delta * J(x_{k+1}) 
                    J_target = U_k + DELTA * J_pred_next

                # Forward pass: Get the network's prediction for the *current* state
                J_pred = model(x_k_batch)
                
                # Calculate the loss, which is the squared error between the prediction and the target 
                loss = loss_function(J_pred, J_target)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.6f}")

        # 4. Save the trained model
        model_path = os.path.join(MODEL_DIR, f"{name.lower()}_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model for {name} saved to {model_path}")