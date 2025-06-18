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
# This now represents the number of shared initial states to generate
NUM_INITIAL_STATES = 5000
TRAJECTORY_LENGTH = 150  # Max steps per trajectory (150 steps * 0.02s/step = 3s)
NUM_EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
# The discount factor delta is specified in the paper
DELTA = 0.5
MODEL_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def generate_shared_start_data(controllers, env, num_initial_states,
                               trajectory_length):
    """
    Generates data by creating a set of random initial states and running
    one trajectory for each controller from each of those shared states.
    """
    print(
        f"Generating data from {num_initial_states} shared initial states...")

    # Initialize a dictionary to hold the transitions for each controller
    datasets = {name: [] for name in controllers.keys()}

    for _ in tqdm(range(num_initial_states)):
        # 1. Generate one shared random initial state
        initial_state = np.array([
            random.uniform(-2.0, 2.0),
            random.uniform(-3.0, 3.0),
            random.uniform(-0.5, 0.5),  # Angle in radians
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

    # The controllers are defined as per the paper
    controllers = {
        # Standard LQR with gains from the paper
        "LQR": LQR(10.0, 12.60, 48.33, 9.09),
        # Sliding Mode controller defined in the paper
        "SM": SlidingMode(env),
        # Velocity Feedback LQR with gains from the paper
        "VF": LQR(0, 30.92, 87.63, 20.40)
    }

    # The cost matrix P is explicitly defined in the paper
    P = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                     dtype=torch.float32)
    p_matrix_device = P.to(DEVICE)

    # --- Generate all data first using the shared-start method ---
    # The paper mentions using 5000 random trajectories for off-line training
    all_datasets = generate_shared_start_data(controllers, env,
                                              NUM_INITIAL_STATES,
                                              TRAJECTORY_LENGTH)

    # --- Main Training Loop for Each Controller ---
    # A separate neural network is trained to evaluate the performance of each controller
    for name, controller_obj in controllers.items():
        print(f"\n{'='*20}\nTraining model for {name} controller\n{'='*20}")

        # 1. Retrieve the pre-generated data for the current controller
        data = all_datasets[name]

        # Prepare data for PyTorch
        x_k_data = torch.tensor([item[0] for item in data],
                                dtype=torch.float32)
        x_k_plus_1_data = torch.tensor([item[1] for item in data],
                                       dtype=torch.float32)

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
            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.6f}")

        # 4. Save the trained model
        model_path = os.path.join(MODEL_DIR, f"{name.lower()}_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model for {name} saved to {model_path}")
