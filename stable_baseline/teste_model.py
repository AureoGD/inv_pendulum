import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Your custom environment
from cont_env.pendulum_env import InvPendulumEnv


def main(args):
    """
    Loads a trained PPO model and plots its performance.
    Assumes the environment provides normalized observations itself.
    """
    # --- 1. Define Paths ---
    run_dir = args.run_dir
    model_path = os.path.join(run_dir, "rl_model_4298624_steps.zip")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No trained model found at: {model_path}")

    # --- 2. Create the Environment (NO VecNormalize) ---
    # Use make_vec_env to ensure it's in the vectorized format SB3 expects.
    eval_env = make_vec_env(lambda: InvPendulumEnv(rendering=args.render), n_envs=1, seed=0)

    # --- 3. Load the Model ---
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, env=eval_env)

    # --- 4. Collect Data for Plotting ---
    print(f"Collecting data from {args.plot_episodes} episodes for plotting...")

    all_episode_rewards = []
    first_episode_data = {"thetas": [], "theta_dots": [], "actions": [], "step_rewards": []}

    for episode in range(args.plot_episodes):
        obs = eval_env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = eval_env.step(action)
            terminated = terminated[0]

            episode_reward += reward[0]

            if episode == 0:
                norm_obs = obs[0]
                theta = np.arctan2(norm_obs[3], norm_obs[2])
                theta_dot = norm_obs[2]

                first_episode_data["thetas"].append(theta)
                first_episode_data["theta_dots"].append(theta_dot)
                first_episode_data["actions"].append(action[0][0])
                first_episode_data["step_rewards"].append(reward[0])

        all_episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")

    eval_env.close()

    # --- 5. Generate Plots ---
    # (Plotting code remains the same as before)
    print("\nGenerating plots...")
    plt.figure("Episode Rewards")
    plt.plot(range(1, args.plot_episodes + 1), all_episode_rewards, 'bo-')
    plt.title('Total Reward per Episode'), plt.xlabel('Episode'), plt.ylabel('Total Reward'), plt.grid(True), plt.show()

    if first_episode_data["thetas"]:
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True, num="Agent Performance")
        time_steps = np.arange(len(first_episode_data["thetas"]))
        fig.suptitle('PPO Agent Performance (First Plotted Episode)', fontsize=16)
        axs[0].plot(time_steps, first_episode_data["thetas"]), axs[0].set_ylabel('Angle (rad)'), axs[0].grid(True)
        axs[1].plot(time_steps, first_episode_data["theta_dots"]), axs[1].set_ylabel('Ang. Velocity'), axs[1].grid(True)
        axs[2].plot(time_steps, first_episode_data["actions"]), axs[2].set_ylabel('Force'), axs[2].grid(True)
        axs[3].plot(time_steps, first_episode_data["step_rewards"]), axs[3].set_xlabel('Time Step'), axs[3].set_ylabel('Reward'), axs[3].grid(True)
        plt.tight_layout(), plt.show()


if __name__ == "__main__":
    # --- Configuration for running with VS Code "Play" button ---
    # IMPORTANT: Replace this with the path to your NEWLY trained model's folder.
    run_dir_to_validate = "nn_models/ppo/27141903"

    from argparse import Namespace
    ide_args = Namespace(run_dir=run_dir_to_validate, plot_episodes=5, render=True)
    main(ide_args)
