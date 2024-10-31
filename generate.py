import torch
import pretty_midi
import numpy as np
from model import LSTMModel

# Load the trained model
def load_model(model_path, input_size=4, hidden_size=128, num_layers=2, output_size=4):
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Generate a sequence of notes
def generate_sequence(model, start_sequence, length=100):
    generated_sequence = start_sequence.tolist()
    input_sequence = torch.tensor(start_sequence, dtype=torch.float32).unsqueeze(0)
    
    for _ in range(length):
        with torch.no_grad():
            output = model(input_sequence)
            next_note = output.squeeze().numpy()
            
            # Add the generated note to the sequence
            generated_sequence.append(next_note)

            # Update the input sequence for the next prediction
            input_sequence = torch.tensor(generated_sequence[-len(start_sequence):], dtype=torch.float32).unsqueeze(0)
    
    return np.array(generated_sequence)

# Save the generated sequence as a MIDI file
def save_to_midi(sequence, output_file="generated_music.mid"):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0) # Acoustic Grand Piano
    duration_scaling_factor = 2.0

    for note_data in sequence:
        start, end, pitch, velocity = note_data

        # Clip pitch and velocity to ensure they are within the MIDI range
        pitch = np.clip(int(pitch * 127), 0, 127)  # Scale and clip pitch
        velocity = np.clip(int(velocity * 127), 0, 127)  # Scale and clip velocity

        # Ensure start and end times are valid
        start = max(0, float(start))
        end = max(start + 0.1, float(end))  # Ensure end is after start

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
    # Example: A short random sequence (adjust values as needed)
    start_sequence = np.random.rand(10, 4)  # 10 notes, 4 features (start, end, pitch, velocity)

    # Generate a sequence of notes
    generated_sequence = generate_sequence(model, start_sequence, length=5000) #modify for the length of the song

    # Save the generated sequence to a MIDI file
    save_to_midi(generated_sequence, output_file="generated_music.mid")
