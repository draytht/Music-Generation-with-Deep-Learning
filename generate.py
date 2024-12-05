# import torch
# import pretty_midi
# import numpy as np
# from model import LSTMModel

# # Load the trained model
# def load_model(model_path, input_size=4, hidden_size=128, num_layers=2, output_size=4):
#     model = LSTMModel(input_size, hidden_size, num_layers, output_size)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()  # Set the model to evaluation mode
#     return model

# # Generate a sequence of notes
# def generate_sequence(model, start_sequence, length=10000000):
#     generated_sequence = start_sequence.tolist()
#     input_sequence = torch.tensor(start_sequence, dtype=torch.float32).unsqueeze(0)
    
#     for _ in range(length):
#         with torch.no_grad():
#             output = model(input_sequence)
#             next_note = output.squeeze().numpy()
            
#             # Add the generated note to the sequence
#             generated_sequence.append(next_note)

#             # Update the input sequence for the next prediction
#             input_sequence = torch.tensor(generated_sequence[-len(start_sequence):], dtype=torch.float32).unsqueeze(0)
    
#     return np.array(generated_sequence)

# # Save the generated sequence as a MIDI file
# def save_to_midi(sequence, output_file="generated_music.mid"):
#     midi = pretty_midi.PrettyMIDI()
#     instrument = pretty_midi.Instrument(program=0) # Acoustic Grand Piano
#     duration_scaling_factor = 2.0

#     for note_data in sequence:
#         start, end, pitch, velocity = note_data

#         # Clip pitch and velocity to ensure they are within the MIDI range
#         pitch = np.clip(int(pitch * 127), 0, 127)  # Scale and clip pitch
#         velocity = np.clip(int(velocity * 127), 0, 127)  # Scale and clip velocity

#         # Ensure start and end times are valid
#         start = max(0, float(start))
#         end = max(start + 0.1, float(end))  # Ensure end is after start

#         # Create a MIDI note and add it to the instrument
#         note = pretty_midi.Note(
#             velocity=velocity,
#             pitch=pitch,
#             start=start,
#             end=end
#         )
#         instrument.notes.append(note)

#     midi.instruments.append(instrument)
#     midi.write(output_file)
#     print(f"Generated MIDI saved as {output_file}")

# # Main function to run the music generation
# if __name__ == "__main__":
#     # Load the model
#     model_path = "music_generation_model.pth"
#     model = load_model(model_path)

#     # Define a starting sequence (seed) for generation
#     # Example: A short random sequence (adjust values as needed)
#     start_sequence = np.random.rand(10, 4)  # 10 notes, 4 features (start, end, pitch, velocity)

#     # Generate a sequence of notes
#     generated_sequence = generate_sequence(model, start_sequence, length=5000) #modify for the length of the song

#     # Save the generated sequence to a MIDI file
#     save_to_midi(generated_sequence, output_file="generated_music.mid")
import torch
import pretty_midi
import numpy as np
from model import LSTMModel

# Load the trained model
def load_model(model_path, input_size=4, hidden_size=128, num_layers=2, output_size=4):
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # Use weights_only=True for security
    model.eval()  # Set the model to evaluation mode
    return model

# Generate a sequence of notes
def generate_sequence(model, start_sequence, length=5000):
    generated_sequence = np.array(start_sequence)  # Convert to numpy array once
    input_sequence = torch.tensor(start_sequence, dtype=torch.float32).unsqueeze(0)
    
    for _ in range(length):
        with torch.no_grad():
            output = model(input_sequence)
            next_note = output.squeeze().numpy()
            
            # Append the generated note to the numpy array
            generated_sequence = np.vstack([generated_sequence, next_note])

            # Update the input sequence for the next prediction
            input_sequence = torch.tensor(generated_sequence[-len(start_sequence):], dtype=torch.float32).unsqueeze(0)
    
    return generated_sequence

# Save the generated sequence as a MIDI file
def save_to_midi(sequence, output_file="generated_music.mid"):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    duration_scaling_factor = 2.0

    for note_data in sequence:
        start, end, pitch, velocity = note_data

        # Clip pitch and velocity to ensure they are within the MIDI range
        pitch = np.clip(int(pitch * 127), 0, 127)  # Scale and clip pitch
        velocity = np.clip(int(velocity * 127), 0, 127)  # Scale and clip velocity

        # Ensure start and end times are valid
        start = max(0, float(start) * duration_scaling_factor)
        end = max(start + 0.1, float(end) * duration_scaling_factor)  # Ensure end is after start

        # Create a MIDI note and add it to the instrument
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=end
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(output_file)
    print(f"Generated MIDI saved as {output_file}")

# Main function to run the music generation
if __name__ == "__main__":
    # Load the model
    model_path = "music_generation_model.pth"
    model = load_model(model_path)

    # Define a starting sequence (seed) for generation
    start_sequence = np.random.rand(10, 4)  # 10 notes, 4 features (start, end, pitch, velocity)

    # Generate a sequence of notes
    generated_sequence = generate_sequence(model, start_sequence, length=5000)  # Set length for desired duration

    # Save the generated sequence to a MIDI file
    save_to_midi(generated_sequence, output_file="C:/Users/Thanh/OneDrive/Desktop/MyProj/Music-Generation-with-Deep-Learning/generate_output/generated_music.mid")



# import torch
# import torch.nn as nn
# import pretty_midi
# import pygame
# import pickle
# import random

# # function to turn the h5 file into a list
# def write_to_list():

#     with open('data.pkl', 'rb') as f:
    
#         loaded_data = pickle.load(f)

#     return loaded_data

# # Define the LSTM model architecture (ensure this matches the saved model)
# class MusicLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MusicLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out)
#         return out

# # Function to load the trained model
# def load_model(model_path, input_size, hidden_size, output_size):
#     model = MusicLSTM(input_size, hidden_size, output_size)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model

# # Function to generate music
# def generate_music(model, seed, length=500):  # Increase the length parameter for longer songs
#     model.eval()
#     generated_sequence = seed.clone().detach().unsqueeze(0).float()
#     with torch.no_grad():
#         for _ in range(length):
#             output = model(generated_sequence)
#             next_note = output[:, -1, :].unsqueeze(1)
#             generated_sequence = torch.cat((generated_sequence, next_note), dim=1)
#     return generated_sequence.squeeze().numpy()

# # Function to convert a sequence to a MIDI file
# def sequence_to_midi(sequence, output_file='generated_music.mid'):
#     midi = pretty_midi.PrettyMIDI()
#     instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
#     for note_info in sequence:
#         start, end, pitch, velocity = note_info
#         note = pretty_midi.Note(
#             velocity=int(velocity),
#             pitch=int(pitch),
#             start=float(start),
#             end=float(end)
#         )
#         instrument.notes.append(note)
#     midi.instruments.append(instrument)
#     midi.write(output_file)

# # Function to pad sequences to the same length and convert to tensor
# def pad_and_convert_to_tensor(sequences):
#     max_length = max(seq.shape[0] for seq in sequences)
#     padded_sbraveequences = [torch.nn.functional.pad(torch.tensor(seq, dtype=torch.float32), (0, 0, 0, max_length - seq.shape[0])) for seq in sequences]
#     tensor = torch.stack(padded_sequences)
#     return tensor

# # Function to play the generated MIDI file
# def play_midi(file_path):
#     pygame.init()
#     pygame.mixer.init()
#     pygame.mixer.music.load(file_path)
#     pygame.mixer.music.play()
#     while pygame.mixer.music.get_busy():
#         pygame.time.Clock().tick(10)

# # Assuming sequences is your list of numpy arrays (processed MIDI sequences)
# sequences = write_to_list()  # Your list of preprocessed MIDI sequences as numpy arrays

# data = pad_and_convert_to_tensor(sequences)

# # Example usage
# input_size = data.shape[2]  # Number of features
# hidden_size = 256
# output_size = data.shape[2]
# model_path = 'new_model.pth'  # Path to your saved model

# # Load the trained model
# model = load_model(model_path, input_size, hidden_size, output_size)
# print("Model loaded successfully.")

# # Use a longer seed sequence for better context
# seed = torch.tensor(sequences[random.randint(0, len(data))][:10]).float()  # Use a longer part of the first sequence as a seed

# # Generate music using the trained model
# generated_sequence = generate_music(model, seed, length=500)  # Generate a longer song
# print("Generated Sequence:", generated_sequence)

# # Save the generated sequence to a MIDI file
# sequence_to_midi(generated_sequence, 'generated_music.mid')

# # Play the generated MIDI file
# play_midi('generated_music.mid')