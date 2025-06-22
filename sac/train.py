import gymnasium as gym
import yaml
import os
import argparse
import importlib
import numpy as np

# Assuming Trainer and BaseAgent are in core
from core.trainer import Trainer
from core.base_agent import BaseAgent  # For type hinting if needed
from cont_env.pendulum_env import InvPendulumEnv
from sac_cas.sac_cas_agent import SacCasAgent


def load_config(config_path="configs/continuous_sac_pendulum.yaml"):  # Default to an example config
    print(config_path)
    """Loads training configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from: {config_path}")
    return config


def get_env_action_spec(env_action_space):
    """Determines action_shape and action_dtype for the replay buffer based on the environment."""
    if isinstance(env_action_space, gym.spaces.Box):  # Continuous
        action_shape = env_action_space.shape
        action_dtype = np.float32
    elif isinstance(env_action_space, gym.spaces.Discrete):  # Discrete
        action_shape = ()  # Scalar integer for action index
        action_dtype = np.int64
    else:
        raise ValueError(f"Unsupported action space type: {type(env_action_space)}")
    return action_shape, action_dtype


def main(args):
    # Load configuration
    config = load_config(args.config)
    env = InvPendulumEnv(rendering=False)
    eval_env = env
    obs_shapes_or_space_for_agent = env.observation_space.shape

    max_grad_norm_from_config = config.get('max_grad_norm', None)  # Default to None
    if isinstance(max_grad_norm_from_config, (int, float)):
        max_grad_norm_val = float(max_grad_norm_from_config)
    else:  # If it's 'null' from YAML (becomes None) or missing
        max_grad_norm_val = None

    # Determine action_spec_for_buffer for the Trainer/Agent
    action_shape, action_dtype = get_env_action_spec(env.action_space)
    action_spec_for_buffer = (action_shape, action_dtype)

    trainer = Trainer(
        agent_class=SacCasAgent,
        env=env,
        eval_env=eval_env,
        training_total_timesteps=int(config.get('training_total_timesteps', 100000)),
        max_steps_per_episode=int(config.get('max_steps_per_episode', 1000)),
        log_interval_timesteps=int(config.get('log_interval_timesteps', 2048)),
        eval_frequency_timesteps=int(config.get('eval_frequency_timesteps', 10000)),
        n_eval_episodes=int(config.get('n_eval_episodes', 5)),
        log_root=config.get('log_root', "runs_dsac"),
        model_root=config.get('model_root', "models_dsac"),
        save_freq_episodes=int(config.get('save_freq_episodes', 100)),
        obs_shapes_or_space=obs_shapes_or_space_for_agent,
        use_encoder=config.get('use_encoder', False),
        encoder_mlp_hidden_dims=config.get('encoder_mlp_hidden_dims', [64, 64]),
        gamma=float(config.get('gamma', 0.99)),
        tau=float(config.get('tau', 0.005)),
        alpha_init=config.get('alpha_init', "auto"),  # Keep as str or float
        critic_lr=float(config.get('critic_lr', 3e-4)),
        actor_lr=float(config.get('actor_lr', 3e-4)),
        replay_buffer_size=int(config.get('replay_buffer_size', 1000000)),
        batch_size=int(config.get('batch_size', 256)),
        learning_starts=int(config.get('learning_starts', 100)),
        gradient_steps=int(config.get('gradient_steps', 1)),
        policy_delay=int(config.get('policy_delay', 2)),
        max_grad_norm=max_grad_norm_val,
        reward_scale=float(config.get('reward_scale', 1.0)),
        action_spec_for_buffer=action_spec_for_buffer,
        aux_data_specs_for_buffer=config.get('aux_data_specs_for_buffer', None)
        # agent_specific_kwargs can be added to config and passed here if needed
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run custom SAC training.")
    parser.add_argument(
        "--config",
        type=str,
        default="sac/configs/cart_pendulum.yaml",  # Default config file
        help="Path to the YAML configuration file for training.")
    args = parser.parse_args()

    main(args)
