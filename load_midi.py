import pretty_midi
import numpy as np
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

# Ignores warnings
warnings.filterwarnings('ignore')

# Function to turn a song into a sequence that could be trained
def midi_to_sequence(file, sequence_length=50):
    try:
        midi_data = pretty_midi.PrettyMIDI(file)
        notes = midi_data.instruments[0].notes
        note_sequence = np.array([[note.start, note.end, note.pitch, note.velocity] for note in notes])
        
        # Standardize sequence length
        if len(note_sequence) < sequence_length:
            padding = np.zeros((sequence_length - len(note_sequence), 4))
            note_sequence = np.vstack((note_sequence, padding))
        else:
            note_sequence = note_sequence[:sequence_length]

        # Normalize features
        note_sequence[:, 0] /= note_sequence[:, 0].max() if note_sequence[:, 0].max() != 0 else 1  # Start
        note_sequence[:, 1] /= note_sequence[:, 1].max() if note_sequence[:, 1].max() != 0 else 1  # End
        note_sequence[:, 2] /= 127  # Pitch (0-127)
        note_sequence[:, 3] /= 127  # Velocity (0-127)
        
        return note_sequence
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

# Method to add all the interpreted files to a list from multiple directories
def files_to_data(directories, sequence_length=50, batch_size=None, max_files=None,  intermediate_save=True): #train model with whole music dataset with max_file=none
    data_list = []
    processed_count = 0

    for directory in directories:
        files = [os.path.join(directory, filename) for filename in os.listdir(directory)]
        print(f"Processing files in directory: {directory}")

        # # Limit to max_files if needed
        # if len(files) > max_files:
        #     files = files[:max_files]
        # Only limit the number of files if max_files is specified
        if max_files is not None and len(files) > max_files:
            files = files[:max_files]

        # Batch processing
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1} in {directory}")

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(midi_to_sequence, file, sequence_length) for file in batch_files]
                
                for future in futures:
                    try:
                        result = future.result()
                        if result is not None:
                            data_list.append(result)
                            processed_count += 1
                            print(f"Processed {processed_count} files so far")
                    except Exception as e:
                        print(f"Error in thread for batch {i // batch_size + 1}: {e}")

            # Intermediate saving after each batch
            if intermediate_save:
                intermediate_filename = f'intermediate_results_{processed_count}.npy'
                save_data(data_list, intermediate_filename)
                print(f"Intermediate save completed: {intermediate_filename}")

        # # Stop if we reached the maximum number of files to process
        # if processed_count >= max_files:
        #     print(f"Reached the maximum limit of {max_files} files. Stopping.")
        #     break

    print("Processing Complete for all directories")
    return data_list

# Function to save data in a binary format
def save_data(data, filename='results.npy'):
    np.save(filename, np.array(data, dtype=object))
    print("Data saved successfully to", filename)