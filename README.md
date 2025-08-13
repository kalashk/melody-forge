# Melody Forge: LSTM Classical Music Generator

This project is an advanced music generation model that uses a Long Short-Term Memory (LSTM) neural network to compose new classical music melodies. The model is trained on the MAESTRO dataset of classical piano performances and is designed to predict a sequence of musical notes (pitch, step, duration) to create new, original pieces.

***

### 🎯 Project Goal

The primary objective is to create a music generation system that can:
* Learn the complex patterns and structures of classical music from a large dataset of MIDI files.
* Generate new musical sequences that are coherent and stylistically similar to the training data.
* Provide a complete workflow from data preprocessing to model training, generation, and MIDI output.

***

### 📚 Dataset

The model is trained on the **MAESTRO (MIDI and Audio Edited for Synchronous Transcription and O) dataset**, a large collection of classical piano performances. The dataset is used to extract note sequences, which are then used to train the LSTM model.

* **Source:** The MAESTRO dataset is downloaded directly from Google's Magenta project storage.

* **Key Data:** The model processes MIDI files, extracting three key features for each note:
    * **Pitch:** The musical note (e.g., C4, D#5).
    * **Step:** The time elapsed since the previous note.
    * **Duration:** The length of the note.

***

### 🧪 Methodology

The project uses a PyTorch-based deep learning pipeline with several key components:

#### 1. Data Preprocessing
* **MIDI Parsing:** The `pretty_midi` library is used to read and parse MIDI files, extracting individual notes from each performance.
* **Quantization:** Note parameters (step and duration) are quantized into predefined bins to simplify the generation task for the model.
* **Tokenization:** Each note is represented as a token of three IDs (pitch_id, step_id, dur_id), which are then used to create sequences for the neural network.
* **Dataset & DataLoader:** A custom `MusicDataset` class is used to prepare sequences of notes and their corresponding target notes, which are then efficiently loaded in batches using a `DataLoader`.

#### 2. Model Architecture
A custom `MusicLSTM_Advanced` model is used, featuring a powerful architecture to capture long-range musical dependencies:
* **Embeddings**: Separate embedding layers for pitch, step, and duration to represent each musical parameter as a dense vector.
* **LSTM Layers**: A deep, multi-layer bidirectional LSTM to process the sequence of musical notes and capture context from both past and future notes.
* **Self-Attention**: A multi-head self-attention mechanism is applied to the LSTM outputs to allow the model to weigh the importance of different notes in the input sequence, improving the coherence of the generated music.
* **Output Heads**: Three separate linear layers predict the pitch, step, and duration of the next note, allowing the model to generate all three parameters independently.

#### 3. Training & Inference
* **Training Loop**: The model is trained using the Adam optimizer and a Cross-Entropy Loss function. Training occurs over several epochs, with the loss being calculated for each of the three output heads.
* **Sampling**: During inference, a combination of **Top-K** and **Nucleus Sampling** is used to generate new notes. This sampling strategy introduces a controlled level of randomness, preventing repetitive or predictable melodies while maintaining musicality.
* **MIDI Generation**: The generated note sequences are decoded and converted back into a MIDI file using `pretty_midi`, allowing the output to be played and listened to.

***

### 🚀 How to Run

1. Clone the MAESTRO dataset:
   ```bash
   !wget [https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip](https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip)
   !unzip maestro-v3.0.0-midi.zip -d maestro_dataset
   ```

2. Run the main Python script in a Colab notebook. Ensure a GPU runtime is selected for faster training.   
   The final output will be a MIDI file named `improved_generated_output.midi` in your Colab environment, which you can download and listen to.
