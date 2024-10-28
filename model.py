import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, output_size=4):
        super(LSTMModel, self).__init__()
        
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define a fully connected layer to map the LSTM output to the target output size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM layer
        out, _ = self.lstm(x)
        
        # Only take the output from the last time step
        out = self.fc(out[:, -1, :])
        
        return out
