# train.py
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import yaml
import numpy as np
import torch
import multiprocessing
from typing import Dict, Any
from functools import partial

# Adjust these import paths to match your project structure
from evolutionary_strategies.cem_optimizer import CEMOptimizer, flatten_nn_parameters, unflatten_parameters_to_state_dict
from evolutionary_strategies.worker import worker_init, run_worker_task, run_dummy_worker_task
from evolutionary_strategies.control_rule import ControlRule
from evolutionary_strategies.training_logs import TrainingLogger


def load_config(config_path="evolutionary_strategies/configs/env_config.yaml") -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at '{config_path}'")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded from {config_path}")
    return config


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Note: Multiprocessing start method could not be set.")

    # --- 1. Load Configuration ---
    config = load_config()
    cem_cfg = config.get('training_config', {}).get('cem', {})
    logger_cfg = config.get('logger_config', {})
    model_cfg = config.get('model_config', {})

    # --- 2. Setup Logger and Prepare Configurations ---
    logger = TrainingLogger(**logger_cfg)

    # --- 3. Instantiate Reference Model and CEM Optimizer ---
    reference_model = ControlRule(observation_dim=4,
                                  output_dim=1,
                                  fc1_dim=model_cfg.get('fc1_dim', 128),
                                  fc2_dim=model_cfg.get('fc2_dim', 128))
    param_dim = flatten_nn_parameters(reference_model).size

    valid_cem_keys = [
        'population_size', 'elite_fraction', 'initial_std_dev', 'update_rule_type', 'elite_weighting_type',
        'noise_decay_factor', 'min_std_dev', 'extra_noise_scale'
    ]
    cem_optimizer_kwargs = {key: cem_cfg[key] for key in valid_cem_keys if key in cem_cfg}
    cem_optimizer = CEMOptimizer(param_dim=param_dim, **cem_optimizer_kwargs)
    cem_optimizer.set_initial_mean_params(reference_model)

    # --- 4. Main Training Loop with a SINGLE, PERSISTENT Pool ---
    num_generations = cem_cfg.get('num_generations', 100)
    population_size = cem_cfg.get('population_size', 100)
    num_workers = min(15, population_size)

    logger.log_message(f"Starting CEM training with {num_workers} persistent parallel workers.")

    # Arguments to initialize each worker process ONCE at the beginning.
    initializer_with_args = partial(worker_init, config=config)

    # Create the pool of workers OUTSIDE the loop
    pool = multiprocessing.Pool(processes=num_workers, initializer=initializer_with_args)

    try:
        for gen in range(1, num_generations + 1):
            # a. Sample a population of NN weight vectors from the CEM
            population_params = cem_optimizer.sample_population()

            # b. Prepare tasks. Each task is just the unique data for that run.
            tasks = [(i, params) for i, params in enumerate(population_params)]

            # c. Map the tasks to the persistent worker pool.
            # The pool reuses the existing processes. No new workers are created here.
            results = pool.map(run_worker_task, tasks)

            # d. Process results
            results.sort(key=lambda x: x[0])
            fitness_scores = [score for _, score in results]

            # e. Update CEM and log
            evaluated_population = list(zip(population_params, fitness_scores))
            cem_optimizer.update_distribution(evaluated_population)
            395,
            logger.log_generation(generation=gen,
                                  evaluated_population=evaluated_population,
                                  cem_optimizer=cem_optimizer,
                                  reference_model_for_saving=reference_model)
    # try:
    #     for gen in range(1, num_generations + 1):
    #         # a. Sample a population (this part stays the same)
    #         population_params = cem_optimizer.sample_population()

    #         # b. Prepare tasks (this part stays the same)
    #         tasks = [(i, params) for i, params in enumerate(population_params)]

    #         # c. --- THIS IS THE ONLY CHANGE ---
    #         #    Map the tasks to the DUMMY worker pool instead of the real one.

    #         # Change this line:
    #         # results = pool.map(run_worker_task, tasks)

    #         # To this:
    #         results = pool.map(run_dummy_worker_task, tasks)

    #         # d. Process results (this part stays the same)
    #         results.sort(key=lambda x: x[0])
    except KeyboardInterrupt:
        logger.log_message("Training interrupted by user.")
    finally:
        logger.log_message("Closing worker pool and saving final model...")

        # --- 5. Clean up the worker pool ---
        pool.close()
        pool.join()

        # ... (save final model and close logger) ...
        final_best_weights = cem_optimizer.get_best_params()
        final_model_state_dict = unflatten_parameters_to_state_dict(final_best_weights, reference_model)
        final_model_path = os.path.join(logger.models_save_dir, "cem_model_final_mean.pth")
        torch.save(final_model_state_dict, final_model_path)
        logger.log_message(f"Final CEM mean weights saved to {final_model_path}")
        logger.close()
