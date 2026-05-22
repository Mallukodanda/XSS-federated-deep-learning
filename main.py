import subprocess
import time
import os
import sys

# --- Configuration ---
NUM_CLIENTS = 5
SERVER_SCRIPT = 'src/server.py'
CLIENT_SCRIPT = 'src/client.py'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def run_fl_simulation():
    """
    Starts the Flower server and clients in separate processes.
    """
    print("--- Starting Federated Learning Simulation ---")
    
    # 1. Start the server
    server_cmd = [sys.executable, SERVER_SCRIPT]
    print(f"Starting server with command: {' '.join(server_cmd)}")
    server_process = subprocess.Popen(server_cmd, cwd=PROJECT_ROOT)
    
    # Give the server a moment to start
    time.sleep(5)
    
    # 2. Start the clients
    client_processes = []
    for i in range(NUM_CLIENTS):
        client_cmd = [sys.executable, CLIENT_SCRIPT, str(i)]
        print(f"Starting client {i} with command: {' '.join(client_cmd)}")
        client_process = subprocess.Popen(client_cmd, cwd=PROJECT_ROOT)
        client_processes.append(client_process)
        # Stagger client start to avoid port conflicts or race conditions
        time.sleep(2) 
        
    # 3. Wait for all processes to finish
    # The server process will terminate after NUM_ROUNDS are completed
    print("\nWaiting for server to complete federated rounds...")
    server_process.wait()
    
    # 4. Terminate client processes (they might still be waiting for the server)
    for p in client_processes:
        if p.poll() is None:
            p.terminate()
            p.wait()
            
    print("\n--- Federated Learning Simulation Finished ---")

if __name__ == '__main__':
    # Ensure the necessary files exist
    if not os.path.exists(os.path.join(PROJECT_ROOT, SERVER_SCRIPT)):
        print(f"Error: Server script not found at {SERVER_SCRIPT}")
        sys.exit(1)
    if not os.path.exists(os.path.join(PROJECT_ROOT, CLIENT_SCRIPT)):
        print(f"Error: Client script not found at {CLIENT_SCRIPT}")
        sys.exit(1)
        
    run_fl_simulation()
