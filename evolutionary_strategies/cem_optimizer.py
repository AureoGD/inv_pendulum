import numpy as np
import torch
from typing import List, Tuple, OrderedDict as OrderedDictType, Dict

# --- Neural Network Weight Helper Functions ---


def flatten_nn_parameters(model: torch.nn.Module) -> np.ndarray:
    """
    Flattens all parameters of a PyTorch model into a single NumPy array.
    """
    return np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])


def unflatten_parameters_to_state_dict(flat_params: np.ndarray,
                                       model_ref: torch.nn.Module) -> OrderedDictType[str, torch.Tensor]:
    """
    Converts a flat NumPy array of parameters back into a PyTorch state_dict.
    Args:
        flat_params (np.ndarray): Flat array of parameters.
        model_ref (torch.nn.Module): A reference model instance to get parameter shapes and names.
    Returns:
        OrderedDict[str, torch.Tensor]: The state dictionary.
    """
    new_state_dict = OrderedDictType()
    current_idx = 0
    for name, param_ref in model_ref.named_parameters():
        num_elements = param_ref.numel()
        shape = param_ref.shape

        # Extract the slice for the current parameter
        param_slice = flat_params[current_idx:current_idx + num_elements]

        # Reshape and convert to tensor
        new_state_dict[name] = torch.from_numpy(param_slice).reshape(shape).float()
        current_idx += num_elements

    if current_idx != flat_params.size:
        raise ValueError(f"Size mismatch: flat_params has {flat_params.size} elements, "
                         f"but model requires {current_idx}.")
    return new_state_dict


class CEMOptimizer:

    def __init__(
            self,
            param_dim: int,
            population_size: int = 50,
            elite_fraction: float = 0.1,
            initial_std_dev: float = 0.1,
            update_rule_type: str = "standard",  # "standard" or "paper_eq3"
            elite_weighting_type: str = "uniform",  # "uniform" or "logarithmic"
            noise_decay_factor: float = 0.995,
            min_std_dev: float = 1e-3,
            extra_noise_scale: float = 0.01):
        if not (0 < elite_fraction <= 1):
            raise ValueError("Elite fraction must be between 0 (exclusive) and 1 (inclusive).")
        if update_rule_type not in ["standard", "paper_eq3"]:
            raise ValueError("update_rule_type must be 'standard' or 'paper_eq3'.")
        if elite_weighting_type not in ["uniform", "logarithmic"]:
            raise ValueError("elite_weighting_type must be 'uniform' or 'logarithmic'.")

        self.param_dim = param_dim
        self.population_size = population_size
        self.num_elites = max(1, int(population_size * elite_fraction))

        self.mean_params = np.zeros(param_dim, dtype=np.float32)
        self.std_devs = np.full(param_dim, initial_std_dev, dtype=np.float32)  # Stores std_devs

        self.update_rule_type = update_rule_type
        self.elite_weighting_type = elite_weighting_type

        self.noise_decay_factor = noise_decay_factor
        self.min_std_dev = min_std_dev
        self.current_extra_noise_scale = extra_noise_scale

        # For "paper_eq3" rule, we need the mean used for sampling the current generation
        self._mu_old_for_current_generation = np.copy(self.mean_params)

        print(
            f"CEMOptimizer initialized: param_dim={param_dim}, pop_size={population_size}, num_elites={self.num_elites}"
        )
        print(f"Update rule: {self.update_rule_type}, Elite weighting: {self.elite_weighting_type}")
        print(f"Initial mean_params: (zeros), Initial std_devs: {initial_std_dev}")

    def set_initial_mean_params(self, initial_model: torch.nn.Module):
        self.mean_params = flatten_nn_parameters(initial_model)
        self._mu_old_for_current_generation = np.copy(self.mean_params)  # Update mu_old as well
        print(f"CEMOptimizer: Initial mean_params set from provided model. Shape: {self.mean_params.shape}")

    def sample_population(self) -> List[np.ndarray]:
        # Store the mean used for this generation's sampling if using paper_eq3 rule
        if self.update_rule_type == "paper_eq3":
            self._mu_old_for_current_generation = np.copy(self.mean_params)

        population = []
        for _ in range(self.population_size):
            individual_params = self.mean_params + self.std_devs * np.random.randn(self.param_dim).astype(np.float32)
            population.append(individual_params)
        return population

    def _calculate_elite_weights(self) -> np.ndarray:
        """Calculates weights for elite individuals based on the configured type."""
        if self.elite_weighting_type == "uniform":
            return np.full(self.num_elites, 1.0 / self.num_elites, dtype=np.float32)

        elif self.elite_weighting_type == "logarithmic":
            # Ranks i = 1, 2, ..., K_e
            ranks = np.arange(1, self.num_elites + 1)
            # log(1 + K_e) / i term from paper (Hansen, 2016)
            # The paper has log(1+K_e)/i, but often log(K_e/2 + 1) - log(i) is used
            # Let's use the paper's direct citation: log( (K_e + 1) / i ) could be problematic if K_e+1 < i
            # A common implementation is: weights_i = log(K_e + 1) - log(i)
            # Or, using the paper's more explicit formula: log(1+K_e)/i part
            raw_weights = np.log(self.num_elites + 1) - np.log(ranks)  # This makes best rank have highest weight
            # Normalize so weights sum to 1
            if np.sum(raw_weights) <= 0:  # Avoid division by zero or negative weights if K_e is small
                return np.full(self.num_elites, 1.0 / self.num_elites, dtype=np.float32)  # Fallback

            weights = raw_weights / np.sum(raw_weights)
            return weights.astype(np.float32)
        else:  # Should not happen due to __init__ check
            return np.full(self.num_elites, 1.0 / self.num_elites, dtype=np.float32)

    def update_distribution(self, evaluated_population: List[Tuple[np.ndarray, float]]):
        if len(evaluated_population) != self.population_size:
            raise ValueError("Size of evaluated_population must match population_size.")

        evaluated_population.sort(key=lambda x: x[1], reverse=True)
        elite_individuals_params = [ind[0] for ind in evaluated_population[:self.num_elites]]

        if not elite_individuals_params:
            print("Warning: No elite individuals selected.")
            return

        elite_params_array = np.array(elite_individuals_params, dtype=np.float32)

        # Calculate elite weights (lambda_i)
        elite_weights = self._calculate_elite_weights()

        # 1. Update mean_params (μ_new) using weighted average of elites
        # This is Equation 1 from the paper.
        self.mean_params = np.average(elite_params_array, axis=0, weights=elite_weights)

        # 2. Update std_devs (or variances)
        if self.update_rule_type == "standard":
            # Weighted standard deviation of elites around their new mean (μ_new)
            # Variance = sum(w_i * (x_i - μ_new)^2)
            # StdDev = sqrt(Variance)
            squared_diffs = np.square(elite_params_array - self.mean_params)
            weighted_variances = np.average(squared_diffs, axis=0, weights=elite_weights)
            self.std_devs = np.sqrt(weighted_variances)

        elif self.update_rule_type == "paper_eq3":
            # Equation 3: Σ_new_diag = Σ λ_i * (z_i - μ_old)²
            # (This directly calculates variance, then we take sqrt for std_dev)
            # _mu_old_for_current_generation was set during sample_population()
            squared_diffs_from_mu_old = np.square(elite_params_array - self._mu_old_for_current_generation)
            new_variances = np.average(squared_diffs_from_mu_old, axis=0, weights=elite_weights)
            self.std_devs = np.sqrt(new_variances)

        # 3. Add decaying extra noise (ε term)
        # The paper adds εI to variance (Σ_new). Adding to std_devs is a common adaptation.
        # If adding to variance: self.std_devs = np.sqrt(np.square(self.std_devs) + self.current_extra_noise_scale)
        # If adding to std_dev directly (simpler for now):
        self.std_devs += self.current_extra_noise_scale  # Adding a base noise level

        # 4. Ensure std_devs do not collapse
        self.std_devs = np.maximum(self.std_devs, self.min_std_dev)

        # 5. Decay the extra noise scale
        self.current_extra_noise_scale *= self.noise_decay_factor
        self.current_extra_noise_scale = max(self.current_extra_noise_scale, self.min_std_dev / 10.0)

        best_fitness_this_gen = evaluated_population[0][1]
        print(f"CEM Distribution Updated. Best Fitness: {best_fitness_this_gen:.4f}, "
              f"Mean StdDev: {np.mean(self.std_devs):.6f}, "
              f"Current Extra Noise Scale: {self.current_extra_noise_scale:.6f}")

    def get_best_params(self) -> np.ndarray:
        return self.mean_params
