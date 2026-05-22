import flwr as fl
import torch
from model import XSSDetector
from central_baseline import load_data, evaluate, DEVICE
from typing import Dict, Optional, Tuple, List
from flwr.common import Scalar
import os

# --- Configuration ---
NUM_CLIENTS = 5
NUM_ROUNDS = 5
MIN_FIT_CLIENTS = 5
MIN_AVAILABLE_CLIENTS = 5
BATCH_SIZE = 64 # Used in load_data for evaluation

# --- Flower Strategy and Server-Side Evaluation ---
def get_evaluate_fn(model: XSSDetector):
    """Return an evaluation function for the server-side evaluation."""
    
    # Load the centralized test set once
    # We use client_id=0 to get the test_loader, as it's the same for all clients
    _, test_loader = load_data(client_id=0)

    def evaluate_fn(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """Evaluate the model on the centralized test set."""
        
        # Set model parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        # Evaluate
        loss, acc, prec, rec, f1 = evaluate(model, test_loader)
        
        print(f"Server-side evaluation round {server_round}: Loss={loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
        
        # Save model checkpoint
        if server_round % 5 == 0:
            torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', f"fl_round_{server_round}.pth"))
            print(f"Saved model checkpoint for round {server_round}.")

        return loss, {"accuracy": acc, "f1_score": f1, "precision": prec, "recall": rec}

    return evaluate_fn

# --- Main Function ---
def start_server():
    print(f"--- Starting Flower Server with {NUM_CLIENTS} clients and {NUM_ROUNDS} rounds ---")
    
    # 1. Initialize the global model
    initial_model = XSSDetector().to(DEVICE)
    
    # 2. Define the strategy
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=MIN_FIT_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        evaluate_fn=get_evaluate_fn(initial_model),
        initial_parameters=fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
        ),
    )
    
    # 3. Start the server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'), exist_ok=True)
    
    # Correct the relative path for load_data in central_baseline.py
    # This is a temporary fix for the server script which runs from src/
    # The load_data function is imported from central_baseline, which has a relative path issue.
    # We will modify the load_data function in central_baseline to use absolute paths or a better relative path.
    # For now, we will assume the server is run from the project root.
    
    # Let's check if the models directory exists in the root
    if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')):
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'), exist_ok=True)
        
    start_server()
