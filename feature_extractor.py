from math import floor, sqrt
from random import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio.transforms as T

import keystroke_extractor as ke


def signal_data_augmentation(signals):
    augmented_signals = []
    for signal in signals:
        augmented_signals.append(signal)
        if random() > 0.5:
            augmented_signals.append(np.roll(signal, int(random() * len(signal) * 0.1)))
        else:
            augmented_signals.append(np.roll(signal, int(random() * -len(signal) * 0.1)))
    return augmented_signals

def mel_spectrogram_data_augmentation(mel_spectrograms):
    time_mask = T.TimeMasking(time_mask_param=len(mel_spectrograms[0]) / 10)
    freq_mask = T.FrequencyMasking(freq_mask_param=len(mel_spectrograms[0]) / 10)
    augmented_mel_spectrograms = []
    for mel_spectrogram in mel_spectrograms:
        augmented_mel_spectrograms.append(mel_spectrogram)
        time_masked = time_mask(torch.tensor(mel_spectrogram))
        freq_masked = freq_mask(time_masked)
        augmented_mel_spectrograms.append(freq_masked.numpy())
    return augmented_mel_spectrograms

def plot_augmented_keystrokes(extracted_keystrokes, samplerate):
    base = sqrt(len(extracted_keystrokes))
    rows = floor(base)
    cols = rows
    while rows * cols < len(extracted_keystrokes):
        cols += 1
        if cols - rows > 1:
            rows += 1
            cols = rows
    
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15), constrained_layout=True)
    for i, keystroke in enumerate(extracted_keystrokes):
        row = i // cols
        col = i % cols
        librosa.display.waveshow(keystroke, sr=samplerate, color="#1f77b4", ax=axs[row, col], max_points=1000)
        axs[row, col].xaxis.set_major_locator(plt.MaxNLocator(3))
        axs[row, col].set_title(f"Keystroke {i+1}")
        axs[row, col].set_xlabel("")
        axs[row, col].set_ylabel("")

    for i in range(len(extracted_keystrokes), rows * cols):
        row = i // cols
        col = i % cols
        fig.delaxes(axs[row, col])

    fig.suptitle("Augmented Keystrokes")
    fig.supxlabel("Time (s)")
    fig.supylabel("Amplitude")
    plt.show()

def generate_mel_spectrogram(signal, samplerate, window_size, hop_size):
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=samplerate, n_mels=64, n_fft=window_size, hop_length=hop_size)
    return mel_spectrogram

def generate_mel_spectrograms(signals, samplerate):
    return [generate_mel_spectrogram(signal, samplerate) for signal in signals]

def display_mel_spectrograms(mel_spectrograms, samplerate, window_size, hop_size):
    base = sqrt(len(mel_spectrograms))
    rows = floor(base)
    cols = rows
    while rows * cols < len(mel_spectrograms):
        cols += 1
        if cols - rows > 1:
            rows += 1
            cols = rows
    
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15), constrained_layout=True)
    for i, mel_spectrogram in enumerate(mel_spectrograms):
        row = i // cols
        col = i % cols
        mel_show = librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis="mel", x_axis="time", ax=axs[row, col], win_length=window_size, hop_length=hop_size, sr=samplerate)
        axs[row, col].set_title("Mel spectrogram " + str(i+1))
        axs[row, col].xaxis.set_major_locator(plt.MaxNLocator(5))
        axs[row, col].set_xlabel("")
        axs[row, col].set_ylabel("")
    
    for i in range(len(mel_spectrograms), rows * cols):
        row = i // cols
        col = i % cols
        fig.delaxes(axs[row, col])

    fig.colorbar(mel_show, ax=axs.ravel().tolist(), format="%+2.0f dB")
    fig.suptitle("Mel Spectrograms")
    fig.supxlabel("Time (s)")
    fig.supylabel("Mels (Hz)")
    plt.show()

if __name__ == "__main__":
    WINDOW_SIZE = 1023
    HOP_SIZE = 225
    BEFORE = int(0.3 * 14400)
    AFTER = int(0.7 * 14400)

    for i in range(0, 26):
        FILE_PATH = f"Recordings/{chr(65 + i)}.wav"  
        signal, samplerate = ke.load_recording(FILE_PATH)
        energy = ke.process_keystrokes(signal, WINDOW_SIZE, HOP_SIZE)
        peaks = ke.isolate_keystroke_peaks(energy)
        keystroke_boundaries = ke.find_keystroke_boundaries(peaks, signal, len(energy), BEFORE, AFTER)

        extracted_keystrokes = ke.isolate_keystrokes(keystroke_boundaries, signal)
        print(f"Extracted Keystrokes: {len(extracted_keystrokes)}")
        ke.plot_extracted_keystrokes(extracted_keystrokes, samplerate)
        
        augmented_keystrokes = signal_data_augmentation(extracted_keystrokes)
        print(f"Augmented Keystrokes: {len(augmented_keystrokes)}")
        plot_augmented_keystrokes(augmented_keystrokes, samplerate)

        mel_spectrograms = [generate_mel_spectrogram(keystroke, samplerate, WINDOW_SIZE, HOP_SIZE) for keystroke in augmented_keystrokes]
        print(f"Mel Spectrograms: {len(mel_spectrograms)}")
        display_mel_spectrograms(mel_spectrograms, samplerate, WINDOW_SIZE, HOP_SIZE)
        
        # save one mel spectrogram to a file
        # np.save("mel_spectrogram.npy", mel_spectrograms[28])

        augmented_mel_spectrograms = mel_spectrogram_data_augmentation(mel_spectrograms)
        print(f"Augmented Mel Spectrograms: {len(augmented_mel_spectrograms)}")
        display_mel_spectrograms(augmented_mel_spectrograms, samplerate, WINDOW_SIZE, HOP_SIZE)