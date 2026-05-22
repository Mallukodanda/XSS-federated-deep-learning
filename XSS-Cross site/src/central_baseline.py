import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import XSSDetector
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import time

# --- Configuration ---
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Utility Functions ---
def load_data(client_id=None):
    """
    Loads the processed data.
    If client_id is None, loads the entire training set (for centralized baseline).
    If client_id is an integer, loads the specific client's data.
    Always loads the centralized test set.
    """
    
    # Load Test Set
    X_test = torch.load(os.path.join(PROCESSED_DATA_DIR, 'X_test_central.pt'))
    y_test = torch.load(os.path.join(PROCESSED_DATA_DIR, 'y_test_central.pt'))
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    if client_id is None:
        # Load all client data for centralized training
        X_train_list, y_train_list = [], []
        for i in range(5): # Assuming 5 clients from data_processing.py
            X_train_list.append(torch.load(os.path.join(PROCESSED_DATA_DIR, f'client_{i}_X.pt')))
            y_train_list.append(torch.load(os.path.join(PROCESSED_DATA_DIR, f'client_{i}_y.pt')))
        
        X_train = torch.cat(X_train_list)
        y_train = torch.cat(y_train_list)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        return train_loader, test_loader
    else:
        # Load specific client data for federated client
        X_train = torch.load(os.path.join(PROCESSED_DATA_DIR, f'client_{client_id}_X.pt'))
        y_train = torch.load(os.path.join(PROCESSED_DATA_DIR, f'client_{client_id}_y.pt'))
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        return train_loader, test_loader # Note: test_loader is the same centralized one

def evaluate(model, data_loader):
    """Evaluates the model on the given data loader."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1

def train_centralized():
    """Trains the XSSDetector model on the combined dataset."""
    print(f"--- Starting Centralized Baseline Training on {DEVICE} ---")
    
    # 1. Load Data
    train_loader, test_loader = load_data(client_id=None)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 2. Initialize Model, Loss, and Optimizer
    model = XSSDetector().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_f1 = 0.0
    
    # 3. Training Loop
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 4. Evaluation
        train_loss = running_loss / len(train_loader)
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.4f} | "
              f"Test F1: {test_f1:.4f}")
              
        # 5. Save best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'central_baseline_best.pth'))
            print("-> Saved best model state.")

    end_time = time.time()
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")
    
    # 6. Final Evaluation
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'central_baseline_best.pth')))
    final_loss, final_acc, final_prec, final_rec, final_f1 = evaluate(model, test_loader)
    
    print("\n--- Final Centralized Baseline Results ---")
    print(f"Loss: {final_loss:.4f}")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"Precision: {final_prec:.4f}")
    print(f"Recall: {final_rec:.4f}")
    print(f"F1-Score: {final_f1:.4f}")
    
    return final_acc, final_f1

if __name__ == '__main__':
    # Create models directory
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'), exist_ok=True)
    train_centralized()
