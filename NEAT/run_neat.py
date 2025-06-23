# run_neat.py

import os
import neat
import numpy as np
import multiprocessing
import pickle
import datetime

# Make sure this import points to your environment file
from cont_env.pendulum_env import InvPendulumEnv


# -----------------------------------------------------------------------------
# FIX: Inherit from neat.reporting.BaseReporter instead of neat.BaseReporter
# -----------------------------------------------------------------------------
class BestGenomeReporter(neat.reporting.BaseReporter):
    """
    A custom NEAT reporter that saves the best genome found so far to a file.
    """

    def __init__(self, max_gen, save_path='.'):
        self.best_fitness = -float('inf')
        self.best_genome = None
        self.save_path = os.path.join(save_path, 'best_genome.pkl')
        self.generation = 0
        self.max_generation = max_gen
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def post_evaluate(self, config, population, species, best_genome):
        """
        This method is called after each generation's evaluation is complete.
        """
        self.generation += 1
        if best_genome and best_genome.fitness is not None:
            if best_genome.fitness > self.best_fitness:
                self.best_fitness = best_genome.fitness
                self.best_genome = best_genome

                print(f"\n--- New best genome found! ---")
                # The 'generation' attribute is on the population object itself
                print(f"Generation: {self.generation}/{self.max_generation}")
                print(f"Fitness: {self.best_fitness:.2f}")

                with open(self.save_path, 'wb') as f:
                    pickle.dump(self.best_genome, f)
                print(f"Saved new best genome to {self.save_path}\n")


# -----------------------------------------------------------------------------


def eval_single_genome(genome, config):
    """
    This function evaluates a single genome.
    It is called by the ParallelEvaluator.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = InvPendulumEnv(env_id='parallel-eval', rendering=False)
    observation, info = env.reset()

    total_reward = 0.0
    for _ in range(env.max_step):
        action_output = net.activate(observation)
        action_numpy = np.array(action_output)
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        total_reward += reward
        if terminated or truncated:
            break

    return total_reward


def run(config_file):
    """
    Sets up and runs the NEAT algorithm using parallel evaluation.
    """
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("sb3", "neat", f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving checkpoints and results to: {run_dir}")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    # --- FIX: Add `neat.reporting.` to all reporter classes ---
    p.add_reporter(neat.reporting.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    checkpoint_prefix = os.path.join(run_dir, 'neat-checkpoint-')
    p.add_reporter(neat.Checkpointer(50, filename_prefix=checkpoint_prefix))

    p.add_reporter(BestGenomeReporter(save_path=run_dir, max_gen=1000))
    # -------------------------------------------------------------
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_single_genome)
    winner = p.run(pe.evaluate, 1000)

    final_winner_path = os.path.join(run_dir, 'final_winner.pkl')
    with open(final_winner_path, 'wb') as f:
        pickle.dump(winner, f)

    print('\nBest genome of the final generation:\n{!s}'.format(winner))


if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat.txt')
    run(config_path)
