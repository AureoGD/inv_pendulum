import os
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Your custom environment
from cont_env.pendulum_env import InvPendulumEnv


def main(args):
    # --- 1. Load Paths and Environment ---
    model_path = args.load_path

    # Check if the model path exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Automatically determine the run directory from the model path
    run_dir = os.path.dirname(model_path)
    print(f"Loading model from: {model_path}")
    print(f"Continuing run in directory: {run_dir}")

    # Create the environment. MUST BE IDENTICAL to the original training setup.
    train_env = make_vec_env(InvPendulumEnv, n_envs=args.n_envs, seed=0)
    eval_env = make_vec_env(InvPendulumEnv, n_envs=1, seed=1)

    # --- 2. Load the Model ---
    # Load the model, and pass the new environment to it
    model = PPO.load(model_path, env=train_env)

    print("\nModel loaded. Continuing training...")

    # --- 3. Setup New Callbacks for the Continued Run ---
    # You can reuse the same callback logic to continue saving the best model and checkpoints
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,  # Save to the same folder
        log_path=run_dir,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=run_dir,
        name_prefix="rl_model_continued",  # Use a new prefix to avoid overwriting
    )

    # --- 4. Continue Training ---
    # The key is `reset_num_timesteps=False`. This tells the model to continue
    # counting timesteps from where the loaded model left off.
    model.learn(
        total_timesteps=args.train_more_steps,
        callback=[eval_callback, checkpoint_callback],
        reset_num_timesteps=False  # IMPORTANT
    )

    # --- 5. Final Save ---
    print("Continued training finished. Saving final model.")
    # Save with a new name to distinguish it from the original final model
    model.save(os.path.join(run_dir, "final_model_continued"))

    print("\n--- Summary ---")
    print(f"Saved final continued model to: {run_dir}/final_model_continued.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-path",
                        type=str,
                        required=True,
                        help="Path to the .zip model file to load and continue training (e.g., nn_models/ppo/RUN_ID/rl_model_500000_steps.zip)")
    parser.add_argument("--train-more-steps", type=int, default=1_000_000, help="Number of additional timesteps to train.")
    # You must provide the SAME n_envs as the original training run
    parser.add_argument("--n-envs", type=int, default=32, help="Number of parallel environments (MUST match original training).")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)

    args = parser.parse_args()
    main(args)
