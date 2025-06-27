import os
import argparse
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# Import both callbacks
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Your custom environment
from cont_env.pendulum_env import InvPendulumEnv


def main(args):
    algo = args.algo.lower()

    # --- 1. Setup Directories ---
    timestamp = datetime.now().strftime("%d%H%M%S")
    run_dir = f"nn_models/{algo}/{timestamp}"
    tensorboard_log = f"nn_tensorboard/{algo}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # --- 2. Create Environments ---
    train_env = make_vec_env(InvPendulumEnv, n_envs=args.n_envs, seed=0)
    # this eval_env is used to evaluate the neural network
    eval_env = make_vec_env(InvPendulumEnv, n_envs=1, seed=1)

    # --- 3. Setup Callbacks ---
    # Callback for saving the best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Save a model every `checkpoint_freq` steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=run_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=False,  # We removed VecNormalize
    )
    # --- End of new section ---

    # --- 4. Create and Train the Model ---
    model = PPO(policy="MlpPolicy", env=train_env, verbose=1, tensorboard_log=tensorboard_log, seed=0)

    print("\nStarting model training...")
    # Pass a list of callbacks to the learn method
    model.learn(
        total_timesteps=args.total_timesteps,
        # Pass both callbacks here
        callback=[eval_callback, checkpoint_callback],
        tb_log_name=timestamp)

    # --- 5. Final Save ---
    print("Training finished. Saving final model.")
    model.save(os.path.join(run_dir, "final_model"))

    # --- 6. Updated Training Summary ---
    print("\n--- Training Summary ---")
    print(f"Saved final model to: {run_dir}/final_model.zip")
    print(f"The best performing model was saved as 'best_model.zip' in the same directory.")
    print(f"Intermediate model checkpoints were saved with the prefix 'rl_model'.")
    print("----------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", help="Algorithm: ppo or sac")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, help="Total timesteps for training")
    parser.add_argument("--n-envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Frequency to run evaluation")

    # --- NEW: Command-line argument for checkpoint frequency ---
    parser.add_argument("--checkpoint-freq",
                        type=int,
                        default=100_000,
                        help="Frequency to save intermediate models (in steps)")
    # --- End of new section ---

    args = parser.parse_args()

    main(args)
