# plot_performance_surfaces.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from neural_network import PerformanceIndexNN

# --- Configuration ---
MODEL_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Plotting Grid Setup ---
# Define ranges for the axes based on the plots in the paper
THETA_RANGE = np.linspace(-1.0, 1.0, 50)  # Pendulum angle (rad)
X_DOT_RANGE = np.linspace(-0.5, 0.5, 50)  # Cart velocity (m/s)
THETA_GRID, X_DOT_GRID = np.meshgrid(THETA_RANGE, X_DOT_RANGE)

# The paper specifies that these plots are slices with x=0 and theta_dot=0 
X_GRID = np.zeros_like(THETA_GRID)
THETA_DOT_GRID = np.zeros_like(THETA_GRID)

# Prepare the grid of state vectors to be fed into the models
# Shape will be (batch_size, 4), where batch_size = 50*50
states_to_predict = np.stack([
    X_GRID.ravel(),
    X_DOT_GRID.ravel(),
    THETA_GRID.ravel(),
    THETA_DOT_GRID.ravel()
], axis=1)
states_tensor = torch.tensor(states_to_predict, dtype=torch.float32).to(DEVICE)


if __name__ == '__main__':
    # --- Check for trained models ---
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Directory '{MODEL_DIR}' not found. Please run train.py first.")
        exit()

    # Define the models to plot
    models_to_plot = [
        ("LQR", "Performance of LQR Controller (Fig. 4)"),
        ("SM", "Performance of Sliding Mode Controller (Fig. 5)"),
        ("VF", "Performance of Velocity Feedback Controller (Fig. 6)")
    ]

    # --- Generate a plot for each model ---
    for name, title in models_to_plot:
        model_path = os.path.join(MODEL_DIR, f"{name.lower()}_model.pth")
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found for controller {name}. Please run train.py.")
            continue

        # Load the trained model
        model = PerformanceIndexNN().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"Loaded model for {name} controller.")

        # Get performance index predictions from the neural network
        with torch.no_grad():
            predicted_cost = model(states_tensor)
        
        # Reshape the flat predictions back into a 2D grid for plotting
        Z_GRID = predicted_cost.cpu().numpy().reshape(THETA_GRID.shape)

        # --- Create the 3D Surface Plot ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # The paper's plots have theta on the x-axis and x_dot on the y-axis
        surf = ax.plot_surface(THETA_GRID, X_DOT_GRID, Z_GRID, cmap='viridis', edgecolor='none')
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(r'$\theta$ (rad)', fontsize=12)
        ax.set_ylabel(r'$\dot{x}$ (m/s)', fontsize=12)
        ax.set_zlabel('Predicted Performance Cost J(x)', fontsize=12)
        fig.colorbar(surf, shrink=0.5, aspect=5, label='Cost')
        
        # Set viewing angle similar to the paper's figures
        ax.view_init(elev=30, azim=-135)
        
        plt.show()