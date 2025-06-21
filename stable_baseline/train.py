import os
import argparse
from datetime import datetime

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cont_env.pendulum_env import InvPendulumEnv
from gymnasium.wrappers import RecordEpisodeStatistics


def main(args):
    algo = args.algo.lower()
    total_timesteps = args.total_timesteps
    n_envs = args.n_envs  # Applies to PPO only
    eval_freq = args.eval_freq
    checkpoint_freq = args.checkpoint_freq

    # Timestamped run folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"sb3/model/{algo}/{timestamp}"
    tensorboard_log = f"sb3/tensorboard/{algo}/{timestamp}"

    os.makedirs(run_dir, exist_ok=True)

    if algo == "ppo":
        # PPO supports multiple envs
        train_env = make_vec_env(InvPendulumEnv, n_envs=n_envs, seed=0)
        # Normalize obs for PPO, not reward by default
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    elif algo == "sac":
        # SAC = always single env
        base_train_env = DummyVecEnv([lambda: InvPendulumEnv()])
        # Normalize obs for SAC (VERY important)
        train_env = VecNormalize(base_train_env,
                                 norm_obs=True,
                                 norm_reward=False)

    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    base_eval_env = DummyVecEnv(
        [lambda: RecordEpisodeStatistics(InvPendulumEnv(max_step=5000))])
    eval_env = VecNormalize(base_eval_env, norm_obs=True, norm_reward=False)

    if algo == "ppo":
        model = PPO(policy="MlpPolicy",
                    env=train_env,
                    verbose=1,
                    tensorboard_log=tensorboard_log,
                    seed=0)

    elif algo == "sac":
        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=run_dir,
        name_prefix="checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True  # Save normalization stats
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=f"{run_dir}/eval_logs",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=total_timesteps,
                callback=[checkpoint_callback, eval_callback],
                log_interval=4)

    model.save(f"{run_dir}/final_model")

    if isinstance(train_env, VecNormalize):
        train_env.save(os.path.join(run_dir, "vec_normalize.pkl"))

    mean_reward, std_reward = evaluate_policy(model,
                                              eval_env,
                                              n_eval_episodes=10,
                                              deterministic=True)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",
                        type=str,
                        default="sac",
                        help="Algorithm: ppo or sac")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs",
                        type=int,
                        default=1,
                        help="Number of parallel envs (PPO only)")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--checkpoint-freq",
                        type=int,
                        default=50_000,
                        help="Frequency to save checkpoints (steps)")
    args = parser.parse_args()

    main(args)
