import gymnasium as gym
import torch
import json
import os
import argparse
import importlib
import time
import numpy as np

from cont_env.pendulum_env import InvPendulumEnv
from sac_cas.sac_cas_agent import SacCasAgent


# This helper function dynamically loads your agent class
def load_agent_class(module_path, class_name):
    """Dynamically imports a class from a given module path."""
    # Assumes the script is run from the directory containing 'dsac'
    # and the module_path is relative to 'dsac' (e.g., "sac_cas.sac_cas_agent")
    full_module_path = f"dsac.{module_path}"
    try:
        module = importlib.import_module(full_module_path)
        AgentClass = getattr(module, class_name)
        return AgentClass
    except ModuleNotFoundError:
        print(f"Error: Could not find the module at '{full_module_path}'.")
        print("Please ensure your PYTHONPATH is set correctly and you are running from the project root.")
        raise
    except AttributeError:
        print(f"Error: Class '{class_name}' not found in module '{full_module_path}'.")
        raise


def get_env_action_spec(env_action_space: gym.Space) -> tuple:
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


def enjoy(args):
    # 1. Load Hyperparameters from the specified run directory
    # The agent was trained with a specific model directory, which contains checkpoints and hparams
    model_dir = args.model_dir
    hyperparameters_path = os.path.join(model_dir, "hyperparameters.json")

    if not os.path.exists(hyperparameters_path):
        print(f"Error: hyperparameters.json not found in the specified directory: {model_dir}")
        return

    with open(hyperparameters_path, 'r') as f:
        all_hparams = json.load(f)

    trainer_hparams = all_hparams['trainer_hyperparameters']
    agent_constructor_params = all_hparams['agent_constructor_hyperparameters_passed']

    env_id = trainer_hparams.get('env_id')
    # agent_module_path and agent_class_name must be in your saved hparams for this to work
    # We will assume they are in trainer_hparams, which is a good practice to add
    agent_module_path = trainer_hparams.get('agent_module')
    agent_class_name = trainer_hparams.get('agent_class_name')

    env = InvPendulumEnv(rendering=True)

    # Reconstruct necessary specs from the fresh environment
    if isinstance(env.observation_space, gym.spaces.Dict):
        obs_shapes_or_space = env.observation_space
    else:
        obs_shapes_or_space = env.observation_space.shape

    action_shape, action_dtype = get_env_action_spec(env.action_space)

    # Override parameters with fresh env info and correct paths
    agent_constructor_params['env'] = env
    agent_constructor_params['obs_shapes_or_space'] = obs_shapes_or_space
    agent_constructor_params['action_shape'] = action_shape
    agent_constructor_params['action_dtype'] = action_dtype
    agent_constructor_params['chkpt_dir'] = model_dir  # Point to the correct checkpoint directory

    # Filter out any params not in the agent's __init__ signature
    valid_agent_params = list(SacCasAgent.__init__.__code__.co_varnames)
    filtered_kwargs_for_agent = {k: v for k, v in agent_constructor_params.items() if k in valid_agent_params}

    print("Instantiating agent...")
    agent = SacCasAgent(**filtered_kwargs_for_agent)

    # 4. Load the trained model weights
    # We load the "best" model saved during evaluation
    print(f"Loading model weights from: {model_dir}")
    agent.load_models(best_model=True)

    # 5. Run evaluation loop
    for episode in range(args.episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step = 0

        while not done and not truncated:
            # Use the deterministic action
            action = agent.choose_action(obs, evaluate=True)

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}, Length = {step}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a trained SAC agent perform in its environment.")
    parser.add_argument("--model-dir",
                        type=str,
                        default='sb3/model/sac/saccas_UnknownEnv_20250622_100347',
                        help="Path to the directory where the trained model and hyperparameters.json are saved.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for rendering the visualization.")
    args = parser.parse_args()

    enjoy(args)
