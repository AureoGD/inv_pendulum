# --- New Class for Advanced Switching Logic ---
class AdvancedSwitcher:
    """
    Implements the advanced switching rule from the second paper (Eq. 4, 5, 6)
    to select a controller while preventing chattering.
    """

    def __init__(self, controller_names):
        """
        Initializes the switcher.
        Args:
            controller_names (list): A list of strings with the names of the controllers.
        """
        self.controller_names = controller_names
        self.reset()

    def select_controller(self, j_predictions):
        """
        Selects the best controller based on the current J_hat predictions
        and the switching rule.

        Args:
            j_predictions (dict): A dictionary mapping controller names to their
                                  current predicted J_hat values.

        Returns:
            str: The name of the selected controller.
        """
        # 1. Determine the set of allowed controllers `I(x_k, L_k)`
        allowed_controllers = set()
        allowed_controllers.add(self.previous_controller_name)  # Always allow the current controller

        for name in self.controller_names:
            if name != self.previous_controller_name:
                # Allow other controllers only if their performance is better than ever before
                if j_predictions[name] < self.L_values[name]:
                    allowed_controllers.add(name)

        # 2. Select the best controller from the ALLOWED set
        allowed_j_predictions = {name: j_predictions[name] for name in allowed_controllers}
        best_controller_name = min(allowed_j_predictions, key=allowed_j_predictions.get)

        return best_controller_name

    def update_state(self, chosen_controller_name, j_predictions):
        """
        Updates the internal state of the switcher (L_values and previous controller)
        after a decision has been made and acted upon.

        Args:
            chosen_controller_name (str): The name of the controller that was selected.
            j_predictions (dict): The dictionary of J_hat predictions from the current step.
        """
        # The L value for the ACTIVE controller is updated if its performance was better
        self.L_values[chosen_controller_name] = min(j_predictions[chosen_controller_name],
                                                    self.L_values[chosen_controller_name])
        self.previous_controller_name = chosen_controller_name

    def reset(self):
        """Resets the switcher's state for a new simulation run."""
        # Initialize L_values to track the best performance seen so far
        self.L_values = {name: float('inf') for name in self.controller_names}
        # Initialize the previously used controller. Start with the first one as default.
        self.previous_controller_name = self.controller_names[0] if self.controller_names else None
