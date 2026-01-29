# Music Generation with Deep Learning

A deep learning project focused on generating original musical compositions. This repository explores the use of Recurrent Neural Networks (RNNs)—specifically LSTMs—to learn the patterns, harmonies, and structures of MIDI-based music to generate new sequences.

## 📌 Project Overview
The goal of this project is to build a generative model that can "compose" music. By training on a dataset of MIDI files (predominantly piano), the model learns the probability distribution of notes and chords to predict and generate subsequent musical elements.

### Key Features
* **Sequential Learning:** Uses Long Short-Term Memory (LSTM) networks to capture long-term dependencies in musical structures.
* **MIDI Processing:** Leverages the `music21` library for parsing, analyzing, and writing MIDI data.
* **Custom Generation:** Ability to seed the model with a musical fragment to generate a unique continuation.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Deep Learning:** TensorFlow / Keras
* **Music Analysis:** Music21
* **Data Handling:** NumPy, Pandas


Bash
pip install music21 tensorflow numpy
2. Training the Model
Place your MIDI files in the /data directory.

Run the preprocessing cells in the notebook to extract notes and chords.

Start the training process. The model will save weights periodically to the /weights folder.

3. Generating Music
Once trained, you can use the generation script/section to create new music:

Load the trained weights.

Provide a "seed" sequence (or let the model pick one randomly).

The model will output a .mid file in the /output folder.

🎵 Results
The generated output can be played using any MIDI player or imported into a DAW (Digital Audio Workstation) like Ableton, FL Studio, or GarageBand for further arrangement.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

Created by draytht


