import argparse
import os
import time

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

from cont_env.pendulum_env import InvPendulumEnv


def main(args):
    algo = args.algo.lower()  # 'ppo' or 'sac'
    model_dir = f"sb3/model/{algo}"
    model_path = os.path.join(model_dir, "2025-06-19_11-35-49/best_model.zip")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No trained model found at: {model_path}")

    # Load model
    if algo == "ppo":
        model = PPO.load(model_path)
    elif algo == "sac":
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    # Create fresh env for testing
    env = InvPendulumEnv(dt=0.002, rendering=True)

    # Evaluate quantitatively
    mean_reward, std_reward = evaluate_policy(model,
                                              env,
                                              n_eval_episodes=args.eval_episodes,
                                              deterministic=True,
                                              render=False)
    print(f"\nMean reward over {args.eval_episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}\n")

    # Visualize qualitatively
    obs, _ = env.reset()
    for step in range(args.render_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # env.render()
        # time.sleep(0.02)  # for smoother rendering
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="sac", help="Algorithm: ppo or sac")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of eval episodes for mean reward")
    parser.add_argument("--render-steps", type=int, default=500, help="Number of steps to render for visualization")
    args = parser.parse_args()

    main(args)
