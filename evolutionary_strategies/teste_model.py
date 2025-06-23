import torch
import numpy as np
import time

# --- Make sure these imports match your project structure ---
from evolutionary_strategies.control_rule import ControlRule
from cont_env.pendulum_env import InvPendulumEnv


def run_test():
    """
    Loads a trained model and runs a single, rendered episode to test its performance.
    """
    # --- 1. Configuration ---
    # IMPORTANT: Update this path to point to your saved model file.
    MODEL_PATH = "sb3/cem/cart_pendulum_20250623-112434/saved_models/overall_best_model.pth"  # e.g., 'saved_models/cem_model_final_mean.pth'

    # Model and environment parameters must match what you used for training.
    OBSERVATION_DIM = 4
    ACTION_DIM = 1
    MODEL_CFG = {'fc1_dim': 256, 'fc2_dim': 256}  # Use the same config as your saved model

    print(f"Loading trained model from: {MODEL_PATH}")

    # --- 2. Load the Saved Weights and Create Instances ---
    try:
        # The training script saved a state_dict, which is the correct way.
        saved_state_dict = torch.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{MODEL_PATH}'")
        print("Please update the MODEL_PATH variable in this script.")
        return

    # Create an instance of your neural network
    model = ControlRule(observation_dim=OBSERVATION_DIM, output_dim=ACTION_DIM, **MODEL_CFG)

    # Load the learned weights into the model instance
    model.load_state_dict(saved_state_dict)

    # IMPORTANT: Set the model to evaluation mode.
    # This disables layers like Dropout or BatchNorm if you were using them.
    model.eval()

    # Create an instance of the environment WITH RENDERING ENABLED
    env = InvPendulumEnv(env_id=0, rendering=True)

    # --- 3. Run the Test Episode ---
    total_reward = 0

    # Get the initial observation from the environment
    # Use a fixed seed for a deterministic, repeatable test
    obs, info = env.reset(seed=42)

    print("\nStarting test run...")
    try:
        # The `with torch.no_grad():` block is crucial for inference.
        # It disables gradient calculations, making the model run much faster.
        with torch.no_grad():
            for i in range(env.max_step):
                # --- The "Agent" Loop ---
                # 1. Convert observation (NumPy array) to a PyTorch Tensor
                obs_tensor = torch.tensor(obs, dtype=torch.float32)

                # 2. Add a batch dimension (models expect batches)
                obs_tensor_batched = obs_tensor.unsqueeze(0)

                # 3. Get the action from the model
                action_tensor = model.forward(obs_tensor_batched)

                # 4. Remove the batch dimension and convert back to a NumPy array
                action_numpy = action_tensor.squeeze(0).cpu().numpy()
                # --- End Agent Loop ---

                # Take a step in the environment with the model's action
                obs, reward, terminated, truncated, info = env.step(action_numpy)

                total_reward += reward

                # Render the current state to the screen
                # The render() method is now called from within step() if rendering is True

                # Add a small delay to make the animation watchable at human speed

                if terminated or truncated:
                    print(f"Episode finished after {i+1} timesteps.")
                    break

        print("\n--- Test Results ---")
        print(f"Total Reward: {total_reward:.2f}")
        print("--------------------")

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        # --- 4. Cleanup ---
        # Always close the environment to shut down the rendering window
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    run_test()
