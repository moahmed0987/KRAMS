# based on: keystroke_extractor.py and feature_extractor.py
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
import scipy.signal


def load_recording(file_path):
    signal, samplerate = librosa.load(file_path, sr=None)
    return signal, samplerate

def plot_waveform(signal, samplerate):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(signal, sr=samplerate, color="#1f77b4", axis="s")
    plt.title("Waveform of Recording")
    plt.ylabel("Amplitude")
    plt.show()

def process_keystrokes(signal, window_size, hop_size):
    energy = []
    for i in range(0, len(signal) - window_size, hop_size):
        windowed_signal = signal[i:i+window_size]
        fft_result = fft(windowed_signal)
        energy.append(np.sum(np.abs(fft_result)))
    energy = np.array(energy)
    energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
    return energy

def plot_energy(energy, samplerate, window_size, hop_size, signal):
    plt.figure(figsize=(10, 6))
    num_windows = (len(signal) - window_size) // hop_size + 1
    midpoints_in_samples = np.arange(window_size / 2, len(signal) - window_size / 2, hop_size)[:num_windows]
    time = midpoints_in_samples / samplerate

    plt.plot(time, energy, label="Normalised Energy")
    plt.ylabel("Energy")
    plt.xlabel("Time (seconds)")
    plt.legend()
    plt.title("Energy of Keystrokes")
    plt.show()

def isolate_keystroke_peaks(energy):
    for i in [x / 100.0 for x in range(1, 101, 1)]:
        peaks, _ = scipy.signal.find_peaks(energy, prominence=i, distance=100)
        if len(peaks) == 25:
            break
    return peaks

def plot_peaks(peaks, energy, signal, window_size, hop_size, samplerate):
    num_windows = (len(signal) - window_size) // hop_size + 1
    midpoints_in_samples = np.arange(window_size / 2, len(signal) - window_size / 2, hop_size)[:num_windows]
    time = midpoints_in_samples / samplerate
    peaks_time = peaks / num_windows * len(signal) / samplerate

    plt.figure(figsize=(10, 6))
    plt.plot(peaks_time, energy[peaks], "x", color="r", label="Peaks")
    plt.plot(time, energy)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Energy")
    plt.show()

def find_keystroke_boundaries(peaks, signal, n_windows, before, after):
    peaks = [int((peak / n_windows) * len(signal)) for peak in peaks]
    keystroke_boundaries = []
    for peak in peaks:
        start = peak - before
        end = peak + after
        keystroke_boundaries.append((start, end))
    return keystroke_boundaries

def plot_keystroke_boundaries(keystrokes, signal, samplerate):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(signal, sr=samplerate, color="#1f77b4", axis="s", label="Waveform")
    keystrokes = [(start / 44100, end / 44100) for start, end in keystrokes]
    for i, (start, end) in enumerate(keystrokes):
        if i == 0:
            plt.axvline(x=start, color="g", linestyle="-", linewidth=0.5, label="Start")
            plt.axvline(x=end, color="m", linestyle="-", linewidth=0.5, label="End")
        else:
            plt.axvline(x=start, color="g", linestyle="-", linewidth=0.5)
            plt.axvline(x=end, color="m", linestyle="-", linewidth=0.5)
    plt.legend()
    plt.show()

def isolate_keystrokes(keystroke_boundaries, signal):
    keystrokes = []
    for start, end in keystroke_boundaries:
        if start < 0:
            keystrokes.append(np.pad(signal[:end], (abs(start), 0)))
        elif end > len(signal):
            keystrokes.append(np.pad(signal[start:], (0, end - len(signal))))
        else:
            keystrokes.append(signal[start:end])
    return keystrokes

def plot_extracted_keystrokes(extracted_keystrokes, samplerate):
    fig, axs = plt.subplots(5, 5, figsize=(15, 20), constrained_layout=True)

    for i, keystroke in enumerate(extracted_keystrokes):
        row = i // 5
        col = i % 5
        librosa.display.waveshow(keystroke, sr=samplerate, color="#1f77b4", ax=axs[row, col], max_points=1000)
        axs[row, col].xaxis.set_major_locator(plt.MaxNLocator(5))
        axs[row, col].set_title(f"Keystroke {i+1}")
        axs[row, col].set_xlabel("")
        axs[row, col].set_ylabel("")

    fig.suptitle("Extracted Keystrokes")
    fig.supxlabel("Time (seconds)")
    fig.supylabel("Amplitude")
    plt.show()

def display_mel_spectrograms(mel_spectrograms, samplerate, window_size, hop_size):
    from math import floor, sqrt
    import matplotlib.ticker as ticker
    base = sqrt(len(mel_spectrograms))
    rows = floor(base)
    cols = rows
    while rows * cols < len(mel_spectrograms):
        cols += 1
        if cols - rows > 1:
            rows += 1
            cols = rows
    titles = ["Original", "Time-masked", "Frequency-masked", "Time and Frequency-masked"]
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15), constrained_layout=True)
    for i, mel_spectrogram in enumerate(mel_spectrograms):
        row = i // cols
        col = i % cols
        mel_show = librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis="mel", x_axis="time", ax=axs[row, col], win_length=window_size, hop_length=hop_size, sr=samplerate, cmap="inferno")
        axs[row, col].set_title(titles[i])
        axs[row, col].set_xlabel("")
        axs[row, col].set_ylabel("")
    
    for i in range(len(mel_spectrograms), rows * cols):
        row = i // cols
        col = i % cols
        fig.delaxes(axs[row, col])

    fig.colorbar(mel_show, ax=axs.ravel().tolist(), format="%+2.0f dB")
    fig.suptitle("Mel Spectrograms")
    fig.supxlabel("Time (s)")
    fig.supylabel("Frequency (Hz)")
    plt.show()


def generate_mel_spectrogram(signal, samplerate, window_size, hop_size):
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=samplerate, n_mels=64, n_fft=window_size, hop_length=hop_size)
    return mel_spectrogram

def mel_spectrogram_data_augmentation(mel_spectrograms):
    import torchaudio.transforms as T
    import torch
    augmented_mel_spectrograms = []
    for mel_spectrogram in mel_spectrograms:
        time_mask = T.TimeMasking(time_mask_param=len(mel_spectrogram) / 10)
        freq_mask = T.FrequencyMasking(freq_mask_param=len(mel_spectrogram) / 10)
        augmented_mel_spectrograms.append(mel_spectrogram)

        time_masked = time_mask(torch.tensor(mel_spectrogram))
        augmented_mel_spectrograms.append(time_masked.numpy())

        freq_masked = freq_mask(torch.tensor(mel_spectrogram))
        augmented_mel_spectrograms.append(freq_masked.numpy())

        # time_freq_masked = freq_mask(time_masked)
        time_freq_masked = np.minimum(freq_masked, time_masked)
        augmented_mel_spectrograms.append(time_freq_masked.numpy())
    return augmented_mel_spectrograms

if __name__ == "__main__":
    WINDOW_SIZE = 1023
    HOP_SIZE = 225
    BEFORE = int(0.2 * 14400)
    AFTER = int(0.8 * 14400)
    RECORDING_DIR = "Recordings"
    FILE_PATH = os.path.join(RECORDING_DIR, "A.wav")
    signal, samplerate = load_recording(FILE_PATH)
    energy = process_keystrokes(signal, WINDOW_SIZE, HOP_SIZE)
    peaks = isolate_keystroke_peaks(energy)
    keystroke_boundaries = find_keystroke_boundaries(peaks, signal, len(energy), BEFORE, AFTER)
    extracted_keystrokes = isolate_keystrokes(keystroke_boundaries, signal)
    to_plot = extracted_keystrokes[22]

    # waveform
    plt.figure()
    librosa.display.waveshow(to_plot, sr=samplerate, color="#1f77b4", max_points=1000)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (a.u.)")
    plt.title("Keystroke 23")
    # vertical lines
    plt.axvline(x=0.04, color="g", linestyle="-", linewidth=0.5, label="\"Press\"")
    plt.axvline(x=0.085, color="g", linestyle="-", linewidth=0.5)
    plt.axvline(x=0.125, color="m", linestyle="-", linewidth=0.5, label="\"Release\"")
    plt.axvline(x=0.15, color="m", linestyle="-", linewidth=0.5)
    plt.legend()
    plt.show()

    # spectrogram
    plt.figure()
    D = librosa.amplitude_to_db(np.abs(librosa.stft(to_plot)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=samplerate)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Keystroke 23")
    plt.show()

    # mel spectrogram
    plt.figure()
    mel_spectrogram = librosa.feature.melspectrogram(y=to_plot, sr=samplerate)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    librosa.display.specshow(mel_spectrogram, y_axis='mel', x_axis='time', sr=samplerate)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Keystroke 23")
    plt.show()

    # signal and time-shifted signal
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    axs[0].plot(to_plot, label="Original")
    axs[0].set_title("Original Keystroke")
    axs[1].plot(np.roll(to_plot, (int(len(to_plot) * 0.1))), label="Time-shifted", color="r")
    axs[1].set_title("Time-shifted Keystroke")
    plt.show()

    # signal and frequency-shifted signal
    mel_spectrogram = generate_mel_spectrogram(to_plot, samplerate, WINDOW_SIZE, HOP_SIZE)
    mel_spectrograms = [mel_spectrogram]
    augmented_mel_spectrograms = mel_spectrogram_data_augmentation(mel_spectrograms)
    display_mel_spectrograms(augmented_mel_spectrograms, samplerate, WINDOW_SIZE, HOP_SIZE)