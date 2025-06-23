# test_neat_winner.py

import os
import neat
import numpy as np
import pickle
import time

from cont_env.pendulum_env import InvPendulumEnv


def test_winner(config_path, winner_path="sb3/neat/run_2025-06-23_14-32-05/best_genome.pkl"):
    """
    Loads the winning genome and runs a rendered simulation.
    """
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultSpeciesSet, neat.DefaultStagnation, neat.DefaultReproduction,
                         config_path)

    # Load the winning genome.
    with open(winner_path, "rb") as f:
        winner = pickle.load(f)

    print("Loaded winning genome.")

    # Create the phenotype (the neural network) from the genome.
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    # Create the environment with rendering enabled.
    env = InvPendulumEnv(env_id='winner-test', rendering=True)
    observation, info = env.reset(seed=0)  # Use a fixed seed for repeatable tests

    total_reward = 0.0
    for i in range(env.max_step):
        env.render()  # Call the render method

        action_output = net.activate(observation)
        action_numpy = np.array(action_output)
        print(action_numpy)
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished after {i+1} timesteps.")
            break

    env.close()
    print(f"Total reward of the winner: {total_reward}")


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat.txt')
    test_winner(config_path)
