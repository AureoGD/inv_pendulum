# train_lbfgs.py

import os
import torch
import numpy as np
import random
from tqdm import tqdm

from model.inverted_pendulum import InvePendulum
from controllers import LQR, SlidingMode
from neural_network import PerformanceIndexNN, calculate_cost_U

# --- Configuration ---
NUM_TRAJECTORIES = 5000
TRAJECTORY_LENGTH = 200
# L-BFGS is a full-batch optimizer, so BATCH_SIZE is not used.
# The number of epochs might need to be higher as each epoch is one single update.
NUM_EPOCHS = 150 
DELTA = 0.5
MODEL_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def generate_data_for_controller(controller, env, num_trajectories, trajectory_length):
    """Generates state transition data (x_k, x_{k+1}) for a given controller."""
    print(f"Generating {num_trajectories} trajectories...")
    transitions = []
    for _ in tqdm(range(num_trajectories)):
        initial_state = np.array([
            random.uniform(-2, 2),
            random.uniform(-2, 2),
            random.uniform(-0.5, 0.5),
            random.uniform(-2, 2)
        ])
        state = env.reset(initial_state=initial_state)

        for _ in range(trajectory_length):
            action = controller.update_control(state)
            next_state = env.step_sim(action)
            transitions.append((state, next_state))
            state = next_state
            if abs(state[2]) > np.pi / 2:
                break
    return transitions

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

    P = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    p_matrix_device = P.to(DEVICE)

    # --- Main Training Loop for Each Controller ---
    for name, controller_obj in controllers.items():
        print(f"\n{'='*20}\nTraining model for {name} controller\n{'='*20}")

        # 1. Generate Data and move the entire dataset to the device
        data = generate_data_for_controller(controller_obj, env, NUM_TRAJECTORIES, TRAJECTORY_LENGTH)
        x_k_full = torch.tensor([item[0] for item in data], dtype=torch.float32).to(DEVICE)
        x_k_plus_1_full = torch.tensor([item[1] for item in data], dtype=torch.float32).to(DEVICE)

        # 2. Setup Model and L-BFGS Optimizer
        model = PerformanceIndexNN().to(DEVICE)
        # L-BFGS is a powerful optimizer that doesn't need a manually tuned learning rate like Adam.
        # A learning rate of 1 is standard as the optimizer performs a line search.
        optimizer = torch.optim.LBFGS(model.parameters(), lr=0.001, max_iter=50)
        loss_function = torch.nn.MSELoss()

        # 3. Training Loop with L-BFGS
        print(f"Starting training for {name} model on {DEVICE} with L-BFGS...")
        model.train()
        for epoch in range(NUM_EPOCHS):
            
            # The L-BFGS optimizer requires a "closure" function.
            # This function clears the gradients, computes the loss, and returns it.
            # The optimizer can call this closure multiple times per step.
            def closure():
                optimizer.zero_grad()
                
                # --- Loss calculation on the FULL batch ---
                with torch.no_grad():
                    J_pred_next = model(x_k_plus_1_full)
                    U_k = calculate_cost_U(x_k_full, p_matrix_device)
                    J_target = U_k + DELTA * J_pred_next

                J_pred = model(x_k_full)
                loss = loss_function(J_pred, J_target)
                
                # The backward pass is called inside the closure.
                loss.backward()
                return loss
            
            # Perform the optimization step.
            loss = optimizer.step(closure)
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.6f}")

        # 4. Save the trained model
        model_path = os.path.join(MODEL_DIR, f"{name.lower()}_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model for {name} saved to {model_path}")