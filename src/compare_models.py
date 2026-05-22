import torch
import os
from model import XSSDetector
from central_baseline import load_data, evaluate, DEVICE

# --- Configuration ---
NUM_ROUNDS = 5 # Should match server.py
CENTRAL_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'central_baseline_best.pth')
FL_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', f'fl_round_{NUM_ROUNDS}.pth')

def compare_models():
    """
    Loads the centralized and federated models and evaluates them on the 
    centralized test set.
    """
    print("--- Comparing Centralized and Federated Models ---")
    
    # 1. Load Test Data
    _, test_loader = load_data(client_id=0)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 2. Initialize Models
    central_model = XSSDetector().to(DEVICE)
    fl_model = XSSDetector().to(DEVICE)
    
    results = {}
    
    # 3. Evaluate Centralized Model
    if os.path.exists(CENTRAL_MODEL_PATH):
        print("\nLoading and evaluating Centralized Baseline Model...")
        central_model.load_state_dict(torch.load(CENTRAL_MODEL_PATH))
        loss, acc, prec, rec, f1 = evaluate(central_model, test_loader)
        results['Centralized'] = {
            'Loss': f"{loss:.4f}",
            'Accuracy': f"{acc:.4f}",
            'Precision': f"{prec:.4f}",
            'Recall': f"{rec:.4f}",
            'F1-Score': f"{f1:.4f}"
        }
    else:
        print(f"\nCentralized model not found at {CENTRAL_MODEL_PATH}. Skipping.")
        results['Centralized'] = {'Error': 'Model not found'}

    # 4. Evaluate Federated Model
    if os.path.exists(FL_MODEL_PATH):
        print("\nLoading and evaluating Federated Learning Model...")
        fl_model.load_state_dict(torch.load(FL_MODEL_PATH))
        loss, acc, prec, rec, f1 = evaluate(fl_model, test_loader)
        results['Federated'] = {
            'Loss': f"{loss:.4f}",
            'Accuracy': f"{acc:.4f}",
            'Precision': f"{prec:.4f}",
            'Recall': f"{rec:.4f}",
            'F1-Score': f"{f1:.4f}"
        }
    else:
        print(f"\nFederated model not found at {FL_MODEL_PATH}. Skipping.")
        results['Federated'] = {'Error': 'Model not found'}
        
    # 5. Print Comparison Table
    print("\n--- Final Model Comparison ---")
    
    # Prepare data for tabulate
    table_data = []
    metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for metric in metrics:
        row = [metric]
        row.append(results.get('Centralized', {}).get(metric, 'N/A'))
        row.append(results.get('Federated', {}).get(metric, 'N/A'))
        table_data.append(row)
        
    from tabulate import tabulate
    print(tabulate(table_data, headers=["Metric", "Centralized", "Federated"], tablefmt="pipe"))
    
    # Save results to a file
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'comparison_results.md')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Model Comparison Results\n\n")
        f.write("## Centralized Baseline Training Summary\n")
        f.write("The centralized model achieved the following results on the centralized test set:\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"| :--- | :--- |\n")
        for k, v in results.get('Centralized', {}).items():
            f.write(f"| {k} | {v} |\n")
        f.write("\n")
        
        f.write("## Federated Learning (FedAvg) Summary\n")
        f.write("The federated model (after 5 rounds) achieved the following results on the centralized test set:\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"| :--- | :--- |\n")
        for k, v in results.get('Federated', {}).items():
            f.write(f"| {k} | {v} |\n")
        f.write("\n")
        
        f.write("## Final Comparison Table\n")
        f.write(tabulate(table_data, headers=["Metric", "Centralized", "Federated"], tablefmt="pipe"))
        f.write("\n")
        
    print(f"\nComparison results saved to {output_path}")

if __name__ == '__main__':
    compare_models()
