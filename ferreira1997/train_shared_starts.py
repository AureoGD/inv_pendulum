# train_lbfgs_minibatch.py

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
NUM_INITIAL_STATES = 5000
TRAJECTORY_LENGTH = 300
NUM_EPOCHS = 50
BATCH_SIZE = 1024
DELTA = 0.5
MODEL_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Configuration for Early Stopping ---
LOSS_STOP_THRESHOLD = 1e-3

print(f"Using device: {DEVICE}")


def generate_shared_start_data(controllers, env, num_initial_states, trajectory_length):
    """
    Generates data by creating a set of random initial states and running
    one trajectory for each controller from each of those shared states.
    """
    print(f"Generating data from {num_initial_states} shared initial states...")
    datasets = {name: [] for name in controllers.keys()}
    for _ in tqdm(range(num_initial_states)):
        initial_state = np.array([
            random.uniform(-1.0, 1.0),
            0,
            random.uniform(-0.4, 0.4),  # Angle in radians
            0
        ])
        for name, controller_obj in controllers.items():
            state = env.reset(initial_state=initial_state)
            for _ in range(trajectory_length):
                action = controller_obj.update_control(state)
                next_state = env.step_sim(action)
                datasets[name].append((state, next_state))
                state = next_state
                if abs(state[2]) > np.pi / 2:
                    break
    return datasets


if __name__ == '__main__':
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    env = InvePendulum()
    controllers = {"LQR": LQR(10.0, 12.60, 48.33, 9.09), "SM": SlidingMode(env), "VF": LQR(0, 30.92, 87.63, 20.40)}

    P = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                     dtype=torch.float32)
    p_matrix_device = P.to(DEVICE)

    all_datasets = generate_shared_start_data(controllers, env, NUM_INITIAL_STATES, TRAJECTORY_LENGTH)

    for name, controller_obj in controllers.items():
        print(f"\n{'='*20}\nTraining model for {name} controller\n{'='*20}")

        data = all_datasets[name]
        x_k_data = torch.tensor([item[0] for item in data], dtype=torch.float32)
        x_k_plus_1_data = torch.tensor([item[1] for item in data], dtype=torch.float32)
        dataset = TensorDataset(x_k_data, x_k_plus_1_data)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = PerformanceIndexNN().to(DEVICE)
        # optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01, max_iter=20)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_function = torch.nn.MSELoss()

        print(f"Starting training for {name} model on {DEVICE} ...")
        model.train()
        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0
            for x_k_batch, x_k_plus_1_batch in dataloader:
                x_k_batch = x_k_batch.to(DEVICE)
                x_k_plus_1_batch = x_k_plus_1_batch.to(DEVICE)

                # def closure():
                #     optimizer.zero_grad()
                #     with torch.no_grad():
                #         J_pred_next = model(x_k_plus_1_batch)
                #         U_k = calculate_cost_U(x_k_batch, p_matrix_device)
                #         J_target = U_k + DELTA * J_pred_next
                #     J_pred = model(x_k_batch)
                #     loss = loss_function(J_pred, J_target)
                #     loss.backward()
                #     return loss

                # loss = optimizer.step(closure)
                optimizer.zero_grad()
                J_pred_next = model(x_k_plus_1_batch).detach()
                U_k = calculate_cost_U(x_k_batch, p_matrix_device)
                J_target = U_k + DELTA * J_pred_next
                J_pred = model(x_k_batch)
                loss = loss_function(J_pred, J_target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.6f}")

            # --- Early Stopping Check ---
            # If the average loss for the epoch is below the threshold, stop training.
            if avg_loss < LOSS_STOP_THRESHOLD:
                print(f"Loss stopping criterion ({LOSS_STOP_THRESHOLD}) met. Finishing training for {name}.")
                break

        # --- End of Training Loop ---

        model_path = os.path.join(MODEL_DIR, f"{name.lower()}_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model for {name} saved to {model_path}")
