from load_midi import files_to_data, save_data
from model import LSTMModel
from train import train
import torch
import torch.nn as nn

# Directories of files
directory = 'C:/Users/Thanh/Downloads/Data processing zip/Data Preprocessing/music/music_just_midi'
small_directory = 'C:/Users/Thanh/Downloads/Data processing zip/Data Preprocessing/music/small_midi'
all_directories = [directory, small_directory]

# Preprocess data with max file limit
data = files_to_data(all_directories, sequence_length=50, batch_size=500, max_files=500, intermediate_save=True)
save_data(data, 'combined_results.npy')

# Set up the model, criterion, and optimizer
model = LSTMModel(input_size=4, hidden_size=128, num_layers=2, output_size=4)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, data, criterion, optimizer, num_epochs=20, batch_size=32)