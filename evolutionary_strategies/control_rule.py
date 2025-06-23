import torch
import torch.nn as nn
import numpy as np


class ControlRule(nn.Module):

    # --- The __init__ method is updated here ---
    def __init__(self, observation_dim: int, output_dim: int, **model_cfg):
        """
        Initializes the ControlRule network.

        Args:
            observation_dim (int): The dimension of the observation space.
            output_dim (int): The dimension of the action space.
            **model_cfg (dict): A dictionary containing model architecture parameters.
                                Expected keys: 'fc1_dim', 'fc2_dim'.
        """
        super().__init__()

        self.observation_dim = observation_dim
        self.output_dim = output_dim

        # Extract layer dimensions from the model_cfg dictionary.
        # Using .get() with a default value makes the class more robust.
        # If the key is not in the config, it will use the default instead of crashing.
        fc1_dim = model_cfg.get('fc1_dim', 128)
        fc2_dim = model_cfg.get('fc2_dim', 128)

        # The NN layers are defined based on the extracted dimensions
        self.fc1 = torch.nn.Linear(observation_dim, fc1_dim)
        self.fc2 = torch.nn.Linear(fc1_dim, fc2_dim)
        self.fc_out = torch.nn.Linear(fc2_dim, output_dim)

    def forward(self, normalized_state_tensor: torch.Tensor) -> torch.Tensor:
        """The forward pass expects a pre-normalized tensor."""
        x = torch.relu(self.fc1(normalized_state_tensor))
        x = torch.relu(self.fc2(x))
        mode_logits = self.fc_out(x)
        return mode_logits
