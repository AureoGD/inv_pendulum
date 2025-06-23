import os
import numpy as np
import torch
from typing import Dict, Any, Tuple

from cont_env.pendulum_env import InvPendulumEnv
from evolutionary_strategies.control_rule import ControlRule
from evolutionary_strategies.cem_optimizer import unflatten_parameters_to_state_dict
import time

sim_instance: InvPendulumEnv = None
control_rule: ControlRule = None


def worker_init(config: Dict[str, Any]):
    global sim_instance, control_rule

    process_id = os.getpid()
    print(f"[Proc {process_id}] Initializing worker...")

    # Pass the relevant part of the config to your environment
    env_cfg = config.get('env_config', {})
    sim_instance = InvPendulumEnv(env_id=process_id, **env_cfg, rendering=False)

    # Pass the relevant part of the config to your control rule
    model_cfg = config.get('model_config', {})
    control_rule = ControlRule(observation_dim=sim_instance.observation_space.shape[0],
                               output_dim=sim_instance.action_space.shape[0],
                               **model_cfg)

    print(f"[Proc {process_id}] Worker initialized successfully.")


def run_worker_task(task_args: Tuple[int, np.ndarray]):
    global sim_instance, control_rule

    if sim_instance is None:
        raise RuntimeError("Worker not initialized correctly.")

    task_id, nn_params_flat = task_args
    try:
        state_dict = unflatten_parameters_to_state_dict(nn_params_flat, control_rule)
        control_rule.load_state_dict(state_dict)
        fitness = 0
        max_step = sim_instance.max_step
        obs, info = sim_instance.reset()
        for step in range(max_step):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            force_tensor = control_rule.forward(obs_tensor)
            force = force_tensor.detach().numpy()
            obs, reward, terminated, truncated, info = sim_instance.step(force)
            fitness += reward
            if terminated or truncated:
                break
        return task_id, fitness

    except Exception as e:
        print(f"[Proc {os.getpid()}, Task {task_id}] ERROR in worker task: {e}")
        import traceback
        traceback.print_exc()
        return task_id, -float('inf')


# --- THIS IS THE DUMMY FUNCTION ---
def run_dummy_worker_task(task_args: Tuple[int, np.ndarray]):
    """
    A dummy task that performs only CPU calculations to test multiprocessing scaling.
    It does NOT use your custom environment (InvPendulumEnv).
    """
    # We can ignore the incoming nn_params and just use the task_id
    task_id, _ = task_args
    val = 0
    try:
        # time.sleep(1)
        # Perform a moderately heavy CPU calculation to simulate the work
        # your environment would do. We'll do some matrix multiplications.
        # This should be 100% parallel-safe.
        a = torch.randn(600, 600)
        b = torch.randn(600, 600)

        # Adjust the loop range so that one run takes about the same time
        # as one run of your real environment (e.g., ~0.2 - 0.3 seconds).
        for _ in range(20):
            c = torch.matmul(a, b)

        # Return a dummy fitness score in the correct format.
        return task_id, 1.0

    except Exception as e:
        # This part is just for safety.
        print(f"[Dummy Worker, Task {task_id}] ERROR: {e}")
        return task_id, -float('inf')
