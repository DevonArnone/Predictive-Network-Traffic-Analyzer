"""
PyTorch LSTM Model for Network Traffic Prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim

class TrafficLSTM(nn.Module):
    """
    LSTM model for predicting network traffic volume
    """
    
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of features per timestep (default: 5)
            hidden_size: Number of LSTM units per layer (default: 64)
            num_layers: Number of LSTM layers (default: 2)
            output_size: Size of output (default: 1 for regression)
            dropout: Dropout rate (default: 0.2)
        """
        super(TrafficLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last timestep
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Fully connected layer
        output = self.fc(last_output)
        
        return output

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train the LSTM model
    
    Args:
        model: LSTM model instance
        X_train: Training sequences
        y_train: Training targets
        X_val: Validation sequences
        y_val: Validation targets
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        
    Returns:
        Trained model and training history
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Mini-batch training
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i + batch_size]
            batch_y = y_train_tensor[i:i + batch_size]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        avg_train_loss = train_loss / (len(X_train_tensor) // batch_size + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    return model, history

