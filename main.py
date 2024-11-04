from load_midi import files_to_data, save_data
from model import LSTMModel
from train import train
import torch
import torch.nn as nn
import time

# Directories of files
directory = 'C:/Users/Thanh/OneDrive/Desktop/MyProj/Music-Generation-with-Deep-Learning/music/music_just_midi'
small_directory = 'C:/Users/Thanh/OneDrive/Desktop/MyProj/Music-Generation-with-Deep-Learning/music/small_midi'
all_directories = [directory, small_directory]

# capture data processing starting time
start_processing_time = time.time()

# Preprocess data with max file limit
data = files_to_data(all_directories, sequence_length=50, batch_size=500, max_files=500, intermediate_save=True)
save_data(data, 'combined_results.npy')

# # Test the model with full dataset
# data = files_to_data(all_directories, sequence_length=50, batch_size=500, intermediate_save=True)
# save_data(data, 'combined_results.npy')

# capture data processing end time
end_processing_time = time.time()
final_processing_time = end_processing_time - start_processing_time
print(f"Final processing time: {final_processing_time: 2f} seconds")

# Set up the model, criterion, and optimizer
model = LSTMModel(input_size=4, hidden_size=128, num_layers=2, output_size=4)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# capture training starting time
start_training_time = time.time()

# Train the model
train(model, data, criterion, optimizer, num_epochs=20, batch_size=32)
# device = torch.device("cuda" if torch.is_available()else "cpu")

# capture trainig end time
end_training_time = time.time()
final_training_time = end_training_time - start_training_time
print(f"Final training time: {final_training_time: 2f} seconds")

# total duration
total_duration = final_processing_time + final_training_time
print(f"Total duration: {total_duration: 2f} seconds")