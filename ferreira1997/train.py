import os
import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
# Assuming you have these files in the specified structure
from model.inverted_pendulum import InvePendulum
from controllers import LQR, SlidingMode
from neural_network import PerformanceIndexNN, calculate_cost_U
from ferreira1997.switching import AdvancedSwitcher
from ferreira1997.plot_sim import SimulationPlotter

# --- Configuration ---
# Data Generation
NUM_TRAIN_TRAJECTORIES = 4950
NUM_VAL_TRAJECTORIES = 50
TRAJECTORY_LENGTH = 250

# Training Params
NUM_EPOCHS_OFFLINE = 30
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
DELTA = 0.5
EARLY_STOPPING_THRESHOLD = 0.002

# Online Sim Params
NUM_ONLINE_EPOCHS = 50
SIMULATION_TIME = 5.0
DT = 0.002

MODEL_DIR = "nn_models/cbnn/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =============================================================================
# DATA GENERATION AND TRAINING FUNCTIONS
# =============================================================================


def generate_dataset(controllers, env, num_trajectories, trajectory_length):
    """Generates a dataset from a specified number of initial states/trajectories."""
    print(f"Generating data from {num_trajectories} initial states...")
    datasets = {name: [] for name in controllers.keys()}
    for _ in tqdm(range(num_trajectories), desc=f"Generating {num_trajectories} trajectories"):
        initial_state = np.array([random.uniform(-2.0, 2.0), random.uniform(-3.0, 3.0), random.uniform(-0.5, 0.5), 0])
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


def offline_training(controllers, p_matrix_device):
    """Performs offline training with a proper train/validation split."""
    env = InvePendulum(dt=DT)
    train_datasets = generate_dataset(controllers, env, NUM_TRAIN_TRAJECTORIES, TRAJECTORY_LENGTH)
    val_datasets = generate_dataset(controllers, env, NUM_VAL_TRAJECTORIES, TRAJECTORY_LENGTH)

    for name in controllers.keys():
        print(f"\n{'='*20}\nOFFLINE Training for {name} controller\n{'='*20}")

        train_data = train_datasets[name]
        train_xk = torch.tensor([item[0] for item in train_data], dtype=torch.float32)
        train_xk1 = torch.tensor([item[1] for item in train_data], dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(train_xk, train_xk1), batch_size=BATCH_SIZE, shuffle=True)

        val_data = val_datasets[name]
        val_xk = torch.tensor([item[0] for item in val_data], dtype=torch.float32)
        val_xk1 = torch.tensor([item[1] for item in val_data], dtype=torch.float32)
        val_loader = DataLoader(TensorDataset(val_xk, val_xk1), batch_size=BATCH_SIZE)

        model = PerformanceIndexNN().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_function = torch.nn.MSELoss()

        for epoch in range(NUM_EPOCHS_OFFLINE):
            model.train()
            train_loss = 0
            for x_k_batch, x_k_plus_1_batch in train_loader:
                x_k_batch, x_k_plus_1_batch = x_k_batch.to(DEVICE), x_k_plus_1_batch.to(DEVICE)
                with torch.no_grad():
                    J_pred_next = model(x_k_plus_1_batch)
                    U_k = calculate_cost_U(x_k_batch, p_matrix_device)
                    J_target = U_k + DELTA * J_pred_next
                J_pred = model(x_k_batch)
                loss = loss_function(J_pred, J_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_k_batch, x_k_plus_1_batch in val_loader:
                    x_k_batch, x_k_plus_1_batch = x_k_batch.to(DEVICE), x_k_plus_1_batch.to(DEVICE)
                    J_pred_next = model(x_k_plus_1_batch)
                    U_k = calculate_cost_U(x_k_batch, p_matrix_device)
                    J_target = U_k + DELTA * J_pred_next
                    J_pred = model(x_k_batch)
                    loss = loss_function(J_pred, J_target)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS_OFFLINE}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            if avg_train_loss < EARLY_STOPPING_THRESHOLD:
                print(f"--- Early stopping as train loss is below {EARLY_STOPPING_THRESHOLD} ---")
                break

        model_path = os.path.join(MODEL_DIR, f"{name.lower()}_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model for {name} saved to {model_path}")


def online_training(controllers, p_matrix_device, num_online_epochs):
    """
    Runs multiple control simulations using the AdvancedSwitcher class.
    """
    print(f"\n{'='*40}\nSTARTING ONLINE SIMULATION (Refactored)\n{'='*40}")

    models = {}
    optimizers = {}
    controller_names = list(controllers.keys())
    for name in controller_names:
        model = PerformanceIndexNN().to(DEVICE)
        model_path = os.path.join(MODEL_DIR, f"{name.lower()}_model.pth")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.train()
        models[name] = model
        optimizers[name] = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE / 10)

    loss_function = torch.nn.MSELoss()
    env = InvePendulum(dt=DT)
    num_steps_per_sim = int(SIMULATION_TIME / DT)

    for epoch in tqdm(range(num_online_epochs), desc="Online Training Epochs"):
        switcher = AdvancedSwitcher(controller_names)
        initial_state = np.array([random.uniform(-2, 2), 0, random.uniform(-0.5, 0.5), 0])
        state = env.reset(initial_state=initial_state)

        for _ in range(num_steps_per_sim):
            current_state_np = state
            current_state_torch = torch.tensor(current_state_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                j_predictions = {name: model(current_state_torch).item() for name, model in models.items()}

            best_controller_name = switcher.select_controller(j_predictions)
            selected_controller = controllers[best_controller_name]
            action = selected_controller.update_control(current_state_np)
            next_state_np = env.step_sim(action)

            model_to_update = models[best_controller_name]
            optimizer = optimizers[best_controller_name]
            next_state_torch = torch.tensor(next_state_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                J_pred_next = model_to_update(next_state_torch)
                U_k = calculate_cost_U(current_state_torch, p_matrix_device)
                J_target = U_k + DELTA * J_pred_next

            J_pred = model_to_update(current_state_torch)
            loss = loss_function(J_pred, J_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            switcher.update_state(best_controller_name, j_predictions)
            state = next_state_np

    print("\nOnline training finished. Running final evaluation run...")
    history = run_final_evaluation(env, models, controllers)
    return history


def run_final_evaluation(env, models, controllers):
    """Runs one last simulation with the refactored switcher for plotting."""
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
    state = env.reset(initial_state=np.array([0, 0, 0.5, 0]))
    num_steps = int(SIMULATION_TIME / DT)

    controller_names = list(controllers.keys())
    switcher = AdvancedSwitcher(controller_names)

    for step in range(num_steps):
        current_state_torch = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            j_predictions = {name: model(current_state_torch).item() for name, model in models.items()}

        best_controller_name = switcher.select_controller(j_predictions)
        selected_controller = controllers[best_controller_name]
        action = selected_controller.update_control(state)
        state = env.step_sim(action)
        switcher.update_state(best_controller_name, j_predictions)

        history['time'].append(step * DT)
        history['x'].append(state[0])
        history['dx'].append(state[1])
        history['angle'].append(state[2])
        history['da'].append(state[3])
        history['active_controller'].append(list(controllers.keys()).index(best_controller_name))
        history['j_lqr'].append(j_predictions['LQR'])
        history['j_sm'].append(j_predictions['SM'])
        history['j_vf'].append(j_predictions['VF'])
        history['action'].append(action)

    return history


if __name__ == '__main__':
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # The env needs to be instantiated here for the SM controller
    env = InvePendulum(dt=DT)
    controllers = {"LQR": LQR(10.0, 12.60, 48.33, 9.09), "SM": SlidingMode(env), "VF": LQR(0, 30.92, 87.63, 20.40)}
    P = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    p_matrix_device = P.to(DEVICE)

    # Offline training
    offline_training(controllers, p_matrix_device)

    # Online trainning
    simulation_history = online_training(controllers, p_matrix_device, num_online_epochs=NUM_ONLINE_EPOCHS)

    # Result
    plotter = SimulationPlotter(list(controllers.keys()))
    plotter.plot(simulation_history)
