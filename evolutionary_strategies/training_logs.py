import os
import csv
import numpy as np
import time
import torch  # Added for saving model state_dict
from typing import List, Dict, Any, Tuple, OrderedDict as OrderedDictType
from torch.utils.tensorboard import SummaryWriter


# Helper function (ideally, this would be imported from cem_optimizer.py or a shared util)
def _unflatten_parameters_to_state_dict_for_logger(flat_params: np.ndarray,
                                                   model_ref: torch.nn.Module) -> OrderedDictType[str, torch.Tensor]:
    """Converts a flat NumPy array of parameters back into a PyTorch state_dict."""
    new_state_dict = OrderedDictType()
    current_idx = 0
    for name, param_ref in model_ref.named_parameters():
        num_elements = param_ref.numel()
        shape = param_ref.shape
        param_slice = flat_params[current_idx:current_idx + num_elements]
        new_state_dict[name] = torch.from_numpy(param_slice).reshape(shape).float()
        current_idx += num_elements
    if current_idx != flat_params.size:
        raise ValueError(f"Size mismatch: flat_params has {flat_params.size} elements, "
                         f"but model_ref requires {current_idx}.")
    return new_state_dict


class TrainingLogger:

    def __init__(self,
                 log_dir: str,
                 experiment_name: str = "cem_experiment",
                 log_to_csv: bool = True,
                 log_to_tensorboard: bool = True,
                 save_best_model: bool = True):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_to_csv = log_to_csv
        self.log_to_tensorboard = log_to_tensorboard if SummaryWriter is not None else False
        self.save_best_model_flag = save_best_model
        self.last_generation_end_time = time.time()

        timestamp_str_file = time.strftime("%Y%m%d-%H%M%S")
        self.run_log_dir = os.path.join(self.log_dir, f"{self.experiment_name}_{timestamp_str_file}")
        os.makedirs(self.run_log_dir, exist_ok=True)

        self.models_save_dir = os.path.join(self.run_log_dir, "saved_models")
        if self.save_best_model_flag:
            os.makedirs(self.models_save_dir, exist_ok=True)

        self.csv_filepath = None
        self.csv_writer = None
        self.csv_file = None
        self.csv_headers = [
            "generation", "timestamp", "generation_duration_sec", "best_fitness_in_gen", "mean_fitness", "std_fitness",
            "min_fitness", "population_size", "overall_best_fitness", "cem_mean_std_dev", "cem_extra_noise_scale"
        ]

        if self.log_to_csv:
            self.csv_filepath = os.path.join(self.run_log_dir, "training_log.csv")
            try:
                self.csv_file = open(self.csv_filepath, 'w', newline='')
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_headers)
                self.csv_writer.writeheader()
                print(f"Logging to CSV: {self.csv_filepath}")
            except IOError as e:
                print(f"Error opening CSV file for logging: {e}")
                self.log_to_csv = False

        self.tb_writer = None
        if self.log_to_tensorboard:
            tb_log_path = os.path.join(self.run_log_dir, "tensorboard_logs")
            try:
                self.tb_writer = SummaryWriter(log_dir=tb_log_path)
                print(f"Logging to TensorBoard: {tb_log_path}")
            except Exception as e:
                print(f"Error initializing TensorBoard SummaryWriter: {e}")
                self.log_to_tensorboard = False

        self.start_time = time.time()
        self.overall_best_fitness = -np.inf

    def save_model_checkpoint(self,
                              generation: int,
                              fitness: float,
                              model_params_flat: np.ndarray,
                              reference_model: torch.nn.Module,
                              is_overall_best: bool = False):
        """Saves the model parameters (weights)."""
        if not self.save_best_model_flag or reference_model is None:
            return

        try:
            state_dict = _unflatten_parameters_to_state_dict_for_logger(model_params_flat, reference_model)

            if is_overall_best:
                filename = "overall_best_model.pth"
                filepath = os.path.join(self.models_save_dir, filename)
            else:
                # Optionally save per-generation best or periodic checkpoints
                filename = f"model_gen_{generation:04d}_fit_{fitness:.2f}.pth"
                filepath = os.path.join(self.models_save_dir, filename)

            torch.save(state_dict, filepath)
            self.log_message(f"Saved model checkpoint to {filepath} (Fitness: {fitness:.4f})")
        except Exception as e:
            self.log_message(f"Error saving model checkpoint: {e}")

    def log_generation(self,
                       generation: int,
                       evaluated_population: List[Tuple[np.ndarray, float]],
                       cem_optimizer=None,
                       reference_model_for_saving: torch.nn.Module = None):  # For saving best model
        """
        Logs metrics for a completed generation and saves the best model if applicable.

        Args:
            generation (int): Current generation number.
            evaluated_population (List[Tuple[np.ndarray, float]]): 
                List of (parameter_vector, fitness_score) for the evaluated population.
            cem_optimizer (CEMOptimizer, optional): CEMOptimizer instance for logging its state.
            reference_model_for_saving (torch.nn.Module, optional): 
                An instance of the NN model structure, used for converting flat params
                to a state_dict before saving. Required if save_best_model_flag is True.
        """
        if not evaluated_population:
            print("Warning: log_generation called with empty evaluated_population.")
            return

        fitness_scores = [score for _, score in evaluated_population]
        if not fitness_scores:  # Should be redundant if evaluated_population is not empty
            print("Warning: No fitness scores found in evaluated_population.")
            return

        best_fitness_in_gen = np.max(fitness_scores)
        mean_fitness = np.mean(fitness_scores)
        std_fitness = np.std(fitness_scores)
        min_fitness = np.min(fitness_scores)
        population_size = len(fitness_scores)

        timestamp_console = time.strftime("%Y-%m-%d %H:%M:%S")
        current_time = time.time()
        generation_duration = current_time - self.last_generation_end_time
        self.last_generation_end_time = current_time
        # Check if this generation's best is the overall best
        new_overall_best_found = False
        if best_fitness_in_gen > self.overall_best_fitness:
            self.overall_best_fitness = best_fitness_in_gen
            new_overall_best_found = True
            # Find parameters of the best individual in this generation
            best_individual_params_in_gen = None
            for params, score in evaluated_population:
                if score == best_fitness_in_gen:
                    best_individual_params_in_gen = params
                    break

            if self.save_best_model_flag and best_individual_params_in_gen is not None and reference_model_for_saving is not None:
                self.save_model_checkpoint(generation,
                                           best_fitness_in_gen,
                                           best_individual_params_in_gen,
                                           reference_model_for_saving,
                                           is_overall_best=True)

        log_entry_csv = {
            "generation": generation,
            "timestamp": timestamp_console,
            "generation_duration_sec": f"{generation_duration:.2f}",
            "best_fitness_in_gen": f"{best_fitness_in_gen:.4f}",
            "mean_fitness": f"{mean_fitness:.4f}",
            "std_fitness": f"{std_fitness:.4f}",
            "min_fitness": f"{min_fitness:.4f}",
            "population_size": population_size,
            "overall_best_fitness": f"{self.overall_best_fitness:.4f}",
            "cem_mean_std_dev": "N/A",
            "cem_extra_noise_scale": "N/A"
        }

        cem_mean_std_dev_val = np.nan
        cem_extra_noise_scale_val = np.nan

        if cem_optimizer:
            if hasattr(cem_optimizer, 'std_devs') and cem_optimizer.std_devs is not None:
                cem_mean_std_dev_val = np.mean(cem_optimizer.std_devs)
                log_entry_csv["cem_mean_std_dev"] = f"{cem_mean_std_dev_val:.6f}"
            if hasattr(cem_optimizer, 'current_extra_noise_scale'):
                cem_extra_noise_scale_val = cem_optimizer.current_extra_noise_scale
                log_entry_csv["cem_extra_noise_scale"] = f"{cem_extra_noise_scale_val:.6f}"

        console_msg_parts = [
            f"Gen: {generation:04d}",
            f"Time: {generation_duration:6.2f}s",  # Added duration
            f"BestFit: {best_fitness_in_gen:.2f}",
            f"MeanFit: {mean_fitness:.2f}",
            f"OverallBest: {self.overall_best_fitness:.2f}"
        ]
        if new_overall_best_found:
            console_msg_parts.append("(NEW BEST!)")

        print(
            " | ".join(console_msg_parts) +
            f" | CEM StdDevMean: {log_entry_csv['cem_mean_std_dev']} | CEM NoiseScale: {log_entry_csv['cem_extra_noise_scale']}"
        )

        if self.log_to_csv and self.csv_writer:
            try:
                self.csv_writer.writerow(log_entry_csv)
                self.csv_file.flush()
            except Exception as e:
                print(f"Error writing to CSV: {e}")

        if self.log_to_tensorboard and self.tb_writer:
            try:
                self.tb_writer.add_scalar("Fitness/Best_in_Generation", best_fitness_in_gen, generation)
                self.tb_writer.add_scalar("Fitness/Mean", mean_fitness, generation)
                self.tb_writer.add_scalar("Fitness/StdDev", std_fitness, generation)
                self.tb_writer.add_scalar("Fitness/Min", min_fitness, generation)
                self.tb_writer.add_scalar("Fitness/Overall_Best", self.overall_best_fitness, generation)
                self.tb_writer.add_scalar("Population/Size", population_size, generation)
                if cem_optimizer:
                    if not np.isnan(cem_mean_std_dev_val):
                        self.tb_writer.add_scalar("CEM/Mean_StdDev_Params", cem_mean_std_dev_val, generation)
                    if not np.isnan(cem_extra_noise_scale_val):
                        self.tb_writer.add_scalar("CEM/Extra_Noise_Scale", cem_extra_noise_scale_val, generation)
                self.tb_writer.flush()
            except Exception as e:
                print(f"Error writing to TensorBoard: {e}")

    def log_message(self, message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        if self.log_to_tensorboard and self.tb_writer:
            try:
                # self.tb_writer.add_text("Messages", message, int(time.time()))
                pass
            except Exception as e:
                print(f"Error writing text to TensorBoard: {e}")

    def close(self):
        if self.csv_file:
            try:
                self.csv_file.close()
                print(f"CSV log file closed: {self.csv_filepath}")
            except Exception as e:
                print(f"Error closing CSV file: {e}")

        if self.tb_writer:
            try:
                self.tb_writer.close()
                print("TensorBoard writer closed.")
            except Exception as e:
                print(f"Error closing TensorBoard writer: {e}")

        end_time = time.time()
        total_duration = end_time - self.start_time
        self.log_message(
            f"Logging finished. Total experiment duration: {time.strftime('%H:%M:%S', time.gmtime(total_duration))}")
