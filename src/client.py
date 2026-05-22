import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from model import XSSDetector
from central_baseline import load_data, evaluate, DEVICE, LEARNING_RATE, NUM_EPOCHS

# --- Flower Client Implementation ---
class XSSClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = XSSDetector().to(DEVICE)
        self.train_loader, self.test_loader = load_data(client_id=client_id)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        print(f"Client {self.client_id} initialized with {len(self.train_loader.dataset)} training samples.")

    def get_parameters(self, config):
        """Returns the current model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Sets the model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Trains the model locally for a number of epochs.
        Returns the updated model parameters, the number of examples, and metrics.
        """
        self.set_parameters(parameters)
        
        self.model.train()
        for epoch in range(NUM_EPOCHS):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        
        # The number of examples is the size of the training dataset
        num_examples = len(self.train_loader.dataset)
        
        # Optionally, evaluate on local test set before returning (not required for FedAvg)
        # loss, acc, _, _, f1 = evaluate(self.model, self.test_loader)
        
        print(f"Client {self.client_id} finished training. Samples: {num_examples}")
        return self.get_parameters(config={}), num_examples, {}

    def evaluate(self, parameters, config):
        """
        Evaluates the model on the local test set.
        Returns the loss, the number of examples, and metrics.
        """
        self.set_parameters(parameters)
        
        loss, acc, prec, rec, f1 = evaluate(self.model, self.test_loader)
        
        num_examples = len(self.test_loader.dataset)
        
        print(f"Client {self.client_id} evaluated. Loss: {loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
        return loss, num_examples, {"accuracy": acc, "f1_score": f1}

# --- Main function to start the client ---
def start_client(client_id):
    # Note: Flower expects the client to be started from the same directory as the script
    # We will use the client_id passed as a command-line argument
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=XSSClient(client_id=client_id).to_client(),
    )

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 client.py <client_id>")
        sys.exit(1)
    
    try:
        client_id = int(sys.argv[1])
        if client_id < 0 or client_id >= 5:
            raise ValueError("Client ID must be between 0 and 4.")
        start_client(client_id)
    except ValueError as e:
        print(f"Invalid argument: {e}")
        sys.exit(1)
