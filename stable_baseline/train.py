import os
import argparse
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from cont_env.pendulum_env import InvPendulumEnv


def main(args):
    algo = args.algo.lower()
    total_timesteps = args.total_timesteps
    n_envs = args.n_envs
    eval_freq = args.eval_freq

    model_dir = f"sb3/model/{algo}"
    tensorboard_log = f"sb3/tensorboard/{algo}"

    os.makedirs(model_dir, exist_ok=True)

    if algo == "ppo":
        # ✅ For gymnasium: no wrapper_class needed, SB3 handles it
        train_env = make_vec_env(
            InvPendulumEnv,
            n_envs=n_envs,
        )
    elif algo == "sac":
        # ✅ Single env for SAC, no wrapper needed
        train_env = InvPendulumEnv()
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    eval_env = InvPendulumEnv()

    if algo == "ppo":
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
        )
    elif algo == "sac":
        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(total_timesteps // 10, 1),
        save_path=model_dir,
        name_prefix="checkpoint",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=f"{model_dir}/eval_logs",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
    )

    model.save(f"{model_dir}/final_model")

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", help="Algorithm: ppo or sac")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel envs (only PPO)")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    args = parser.parse_args()

    main(args)
