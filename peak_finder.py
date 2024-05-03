import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.fft import fft


def load_recording(file_path):
    signal, samplerate = librosa.load(file_path, sr=None)
    return signal, samplerate

def process_keystrokes(signal, window_size, hop_size):
    energy = []
    for i in range(0, len(signal) - window_size, hop_size):
        windowed_signal = signal[i:i+window_size]
        fft_result = fft(windowed_signal)
        energy.append(np.sum(np.abs(fft_result)))
    energy = np.array(energy)
    energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
    return energy

def isolate_keystroke_peaks(energy):
    for i in [x / 100.0 for x in range(1, 101, 1)]:
        peaks, _ = scipy.signal.find_peaks(energy, prominence=i)
        if len(peaks) == 25:
            break
    if len(peaks) != 25:
        print("Peaks not found.")
        print()
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


min_diff = np.inf
for i in range(0, 26):
    FILE_PATH = os.path.join("Recordings", chr(65+i)+".wav")
    WINDOW_SIZE = 1023
    HOP_SIZE = 225
    signal, samplerate = load_recording(FILE_PATH)
    energy = process_keystrokes(signal, WINDOW_SIZE, HOP_SIZE)
    
    print(FILE_PATH)
    peaks = isolate_keystroke_peaks(energy)

    if len(peaks) != 0:
        plot_peaks(peaks, energy, signal, WINDOW_SIZE, HOP_SIZE, samplerate)
        print("np.diff(peaks):", np.diff(peaks))
        print("min(np.diff(peaks)):", min(np.diff(peaks)))
        if input("Correctly identified peaks? (y/n): ") == "n":
            print()
            continue
        if min(np.diff(peaks)) < min_diff:
            min_diff = min(np.diff(peaks))
        print()
print("Minimum difference between peaks:", min_diff)