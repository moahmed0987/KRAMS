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
def plot_energy(energy, threshold):
    plt.figure(figsize=(10, 6))
    plt.plot(energy, label='Normalized Energy')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Frame')
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