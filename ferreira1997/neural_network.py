# neural_network.py

import torch
import torch.nn as nn

class PerformanceIndexNN(nn.Module):
    """
    Implements the neural network architecture from Figure 2 of the paper.
    It approximates the cost-to-go function J(x).
    Architecture: Two hidden layers with tanh, plus a linear skip connection.
    """
    def __init__(self, input_size=4, hidden_size1=256, hidden_size2=256):
        super().__init__()
        
        self.hidden_path = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.Tanh(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.Tanh(),
            nn.Linear(hidden_size2, 1)
        )
        self.skip_connection = nn.Linear(input_size, 1)

    def forward(self, x):
        output = self.hidden_path(x) + self.skip_connection(x)
        return output

P = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 5.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
], dtype=torch.float32)

# --- FIX IS HERE ---
# This new implementation is memory-efficient and avoids the large matrix multiplication.
def calculate_cost_U(x, p_matrix):
    """
    Calculates the pointwise cost U(x) = x^T*P*x for a batch of states x
    in a memory-efficient way.
    """
    # Step 1: Calculate x @ P for each row in the batch.
    x_p = torch.matmul(x, p_matrix)  # Shape: (batch_size, 4)
    
    # Step 2: Calculate the dot product of each row of x_p with the corresponding row of x.
    # This is equivalent to getting the diagonal of (x @ P @ x.T) without computing
    # the full matrix.
    cost = torch.sum(x_p * x, dim=1) # Shape: (batch_size,)
    
    return cost.unsqueeze(1) # Reshape to (batch_size, 1)


if __name__ == '__main__':
    model = PerformanceIndexNN()
    print("Neural network model created:")
    print(model)
    
    dummy_input = torch.randn(2, 4)
    output = model(dummy_input)
    print("\nDummy input shape:", dummy_input.shape)
    print("Dummy output shape:", output.shape)