import os

import numpy as np
import pandas as pd

import feature_extractor as fe
import keystroke_extractor as ke

def get_file_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]

def process_recordings(file_paths, window_size, hop_size, before, after):
    df_relative_paths = []
    df_labels = []
    df_mel_spectrograms = []
    for i, file_path in enumerate(file_paths):
        print(f"Processing {file_path}")
        augmented_mel_spectrograms = data_processing_pipeline(file_path, WINDOW_SIZE, HOP_SIZE, BEFORE, AFTER)
        for mel_spectrogram in augmented_mel_spectrograms:
            df_relative_paths.append(file_path)
            df_labels.append(chr(65 + i))
            df_mel_spectrograms.append(mel_spectrogram)
    return df_relative_paths, df_labels, df_mel_spectrograms

def data_processing_pipeline(file_path, window_size, hop_size, before, after):
    signal, samplerate = ke.load_recording(file_path)
    energy = ke.process_keystrokes(signal, window_size, hop_size)
    peaks = ke.isolate_keystroke_peaks(energy)
    keystroke_boundaries = ke.find_keystroke_boundaries(peaks, signal, len(energy), before, after)
    extracted_keystrokes = ke.isolate_keystrokes(keystroke_boundaries, signal)
    augmented_keystrokes = fe.signal_data_augmentation(extracted_keystrokes)
    mel_spectrograms = [fe.generate_mel_spectrogram(keystroke, samplerate, window_size, hop_size) for keystroke in augmented_keystrokes]
    augmented_mel_spectrograms = fe.mel_spectrogram_data_augmentation(mel_spectrograms)
    return augmented_mel_spectrograms

def to_dataframe(df_relative_paths, df_labels, df_mel_spectrograms):
    df = pd.DataFrame(columns=["id", "relative_path", "label", "mel_spectrogram"])
    df["id"] = range(1, len(df_relative_paths) + 1)
    df["relative_path"] = df_relative_paths
    df["label"] = df_labels
    df["mel_spectrogram"] = df_mel_spectrograms
    return df

def to_csv(spectrogram_df, csv_file_path):
    with np.printoptions(threshold=np.inf):
        spectrogram_df.to_csv(csv_file_path, index=False)

if __name__ == "__main__":
    WINDOW_SIZE = 1024
    HOP_SIZE = 225
    BEFORE = int(0.2 * 14400)
    AFTER = int(0.8 * 14400)
    DIRECTORY = "Recordings"
    file_paths = get_file_paths(DIRECTORY)
    df_relative_paths, df_labels, df_mel_spectrograms = process_recordings(file_paths, WINDOW_SIZE, HOP_SIZE, BEFORE, AFTER)
    df = to_dataframe(df_relative_paths, df_labels, df_mel_spectrograms)
    to_csv(df, "keystroke_data.csv")