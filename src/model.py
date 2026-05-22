import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Configuration ---
VOCAB_SIZE = 256  # Character-level tokenization (ASCII)
EMBEDDING_DIM = 128
CNN_FILTERS = 128
KERNEL_SIZE = 3
LSTM_HIDDEN_DIM = 128
NUM_CLASSES = 2
DROPOUT_RATE = 0.5

class Attention(nn.Module):
    """
    Attention mechanism for the output of the LSTM.
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn_weights = nn.Parameter(torch.Tensor(hidden_size, 1))
        nn.init.xavier_uniform_(self.attn_weights.data)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        
        # 1. Calculate attention scores (energy)
        # score = tanh(lstm_output @ attn_weights)
        # score shape: (batch_size, seq_len, 1)
        score = torch.tanh(torch.matmul(lstm_output, self.attn_weights))
        
        # 2. Calculate attention weights (alpha)
        # alpha shape: (batch_size, seq_len, 1)
        alpha = F.softmax(score, dim=1)
        
        # 3. Apply attention weights to LSTM output
        # context = sum(alpha * lstm_output)
        # context shape: (batch_size, hidden_size)
        context = torch.sum(alpha * lstm_output, dim=1)
        
        return context, alpha

class XSSDetector(nn.Module):
    def __init__(self):
        super(XSSDetector, self).__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, padding_idx=0)
        
        # 2. 1D Convolutional Layer
        self.conv1d = nn.Conv1d(
            in_channels=EMBEDDING_DIM, 
            out_channels=CNN_FILTERS, 
            kernel_size=KERNEL_SIZE, 
            padding=1
        )
        
        # 3. LSTM Layer
        # The input size to LSTM is the output of CNN (CNN_FILTERS)
        self.lstm = nn.LSTM(
            input_size=CNN_FILTERS, 
            hidden_size=LSTM_HIDDEN_DIM, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 4. Attention Mechanism
        # The hidden size is 2 * LSTM_HIDDEN_DIM because of bidirectionality
        self.attention = Attention(2 * LSTM_HIDDEN_DIM)
        
        # 5. Fully Connected Layer
        self.fc = nn.Linear(2 * LSTM_HIDDEN_DIM, NUM_CLASSES)
        
        # 6. Dropout
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        
        # 1. Embedding
        x = self.embedding(x)
        # x shape: (batch_size, seq_len, EMBEDDING_DIM)
        
        # 2. CNN (requires input shape: (batch_size, EMBEDDING_DIM, seq_len))
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1d(x))
        # x shape: (batch_size, CNN_FILTERS, seq_len)
        
        # 3. LSTM (requires input shape: (batch_size, seq_len, CNN_FILTERS))
        x = x.permute(0, 2, 1)
        
        # LSTM output: (output, (h_n, c_n))
        # output shape: (batch_size, seq_len, 2 * LSTM_HIDDEN_DIM)
        lstm_output, _ = self.lstm(x)
        
        # 4. Attention
        # context shape: (batch_size, 2 * LSTM_HIDDEN_DIM)
        context, _ = self.attention(lstm_output)
        
        # 5. Dropout
        x = self.dropout(context)
        
        # 6. Fully Connected Layer
        # output shape: (batch_size, NUM_CLASSES)
        logits = self.fc(x)
        
        return logits

if __name__ == '__main__':
    # Example usage and shape check
    model = XSSDetector()
    
    # Dummy input: batch_size=64, seq_len=100
    dummy_input = torch.randint(1, VOCAB_SIZE, (64, 100))
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Model: {model}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Expected output shape: (64, 2)
    assert output.shape == (64, NUM_CLASSES)
    print("Shape check passed.")
