import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.fft import fft


# Load recording of keystroke training data
def load_recording(file_path):
    signal, samplerate = sf.read(file_path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    return signal, samplerate

# Plot the waveform of the recording
def plot_waveform(signal, samplerate):
    plt.figure(figsize=(10, 4))  # Set the figure size
    time = [float(i)/samplerate for i in range(len(signal))]  # Calculate time from samples
    plt.plot(time, signal)
    plt.title('Waveform of Recording')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.show()

# Convert the signal to the frequency domain and normalise energy
def process_keystrokes(signal, window_size, hop_size):
    energy = []
    for i in range(0, len(signal) - window_size, hop_size):
        windowed_signal = signal[i:i+window_size]
        fft_result = fft(windowed_signal)
        energy.append(np.sum(np.abs(fft_result)))
    energy = np.array(energy)
    energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
    return energy

# Plot the energy and threshold to visualize keystrokes
def plot_energy(energy, samplerate, threshold,window_size, hop_size):
    plt.figure(figsize=(10, 6))
    num_windows = (len(signal) - window_size) // hop_size + 1
    midpoints_in_samples = np.arange(window_size / 2, len(signal) - window_size / 2, hop_size)[:num_windows]
    time = midpoints_in_samples / samplerate

    plt.plot(time, energy, label='Normalized Energy')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.legend()
    plt.title('Keystroke Detection')
    plt.show()

# Isolate keystrokes based on the threshold and energy values
def isolate_keystrokes(energy, threshold):
    keystroke_indices = np.where(energy > threshold)[0]
    keystrokes = []
    current_index = keystroke_indices[0]

    for i in range(1, len(keystroke_indices)):
        if keystroke_indices[i] != keystroke_indices[i-1] + 1:
            if current_index != keystroke_indices[i-1]:
                keystrokes.append((current_index, keystroke_indices[i-1]))
            current_index = keystroke_indices[i]
    if current_index != keystroke_indices[-1]:
        keystrokes.append((current_index, keystroke_indices[-1]))
    return keystrokes

# Plot the keystrokes on the energy graph
def plot_keystrokes(keystrokes):
    for start, end in keystrokes:
        plt.axvline(x=start, color='g', linestyle='-', linewidth=0.5)
        plt.axvline(x=end, color='m', linestyle='-', linewidth=0.5)
    plt.show()

def extract_keystrokes(keystrokes, signal, windows):
    extracted_keystrokes = []
    for start, end in keystrokes:
        start_time = int((start / windows) * len(signal))
        end_time = int((end / windows) * len(signal))
        extracted_keystrokes.append(signal[start_time:end_time])
            
    return extracted_keystrokes

def plot_extracted_keystrokes(extracted_keystrokes, samplerate):
    n_keystrokes = len(extracted_keystrokes)
    n_rows = (n_keystrokes + 3) // 4
    fig, axs = plt.subplots(n_rows, 4, figsize=(15, 4*n_rows))

    for i, keystroke in enumerate(extracted_keystrokes):
        row = i // 4
        col = i % 4
        time = [float(i)/samplerate for i in range(len(keystroke))]
        axs[row, col].plot(time, keystroke)
        axs[row, col].set_title(f'Keystroke {i+1}')
        axs[row, col].set_xlabel('Time [s]')
        axs[row, col].set_ylabel('Amplitude')

    for i in range(n_keystrokes, n_rows * 4):
        row = i // 4
        col = i % 4
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.show()

window_size = 10000
hop_size = window_size // 2
threshold = 0.3
file_path = 'Recordings\B.wav'

signal, samplerate = load_recording(file_path)
plot_waveform(signal, samplerate)
energy = process_keystrokes(signal, window_size, hop_size)
plot_energy(energy, samplerate, threshold, window_size, hop_size)
keystrokes = isolate_keystrokes(energy, threshold)
plot_keystrokes(keystrokes)
extracted_keystrokes = extract_keystrokes(keystrokes, signal, len(energy))
plot_extracted_keystrokes(extracted_keystrokes, samplerate)