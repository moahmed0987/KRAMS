import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
import scipy.signal


# Load recording of keystroke training data
def load_recording(file_path):
    signal, samplerate = librosa.load(file_path, sr=None)
    return signal, samplerate

# Plot the waveform of the recording
def plot_waveform(signal, samplerate):
    plt.figure(figsize=(10, 4))  # Set the figure size
    librosa.display.waveshow(signal, sr=samplerate, color='#1f77b4', axis="s")
    plt.title('Waveform of Recording')
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
def plot_energy(energy, samplerate, threshold, window_size, hop_size):
    plt.figure(figsize=(10, 6))
    num_windows = (len(signal) - window_size) // hop_size + 1
    midpoints_in_samples = np.arange(window_size / 2, len(signal) - window_size / 2, hop_size)[:num_windows]
    time = midpoints_in_samples / samplerate

    plt.plot(time, energy, label='Normalised Energy')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.ylabel('Energy')
    plt.legend()
    plt.title('Keystroke Detection')
    plt.show()

def isolate_keystroke_peaks(energy):
    for i in [x / 100.0 for x in range(1, 101, 1)]:
        peaks, _ = scipy.signal.find_peaks(energy, prominence=i)
        if len(peaks) == 25:
            break
    return peaks

def plot_peaks(peaks, energy):
    plt.figure(figsize=(10, 6))
    plt.plot(peaks, energy[peaks], "x", color='r', label='Peaks')
    plt.plot(energy)
    plt.show()

def find_keystroke_boundaries(peaks, signal, n_windows, before, after):
    peaks = [int((peak / n_windows) * len(signal)) for peak in peaks]
    keystroke_boundaries = []
    for peak in peaks:
        start = peak - before if peak - before > 0 else 0
        end = peak + after if peak + after < len(signal) else len(signal)
        keystroke_boundaries.append((start, end))
    return keystroke_boundaries

def plot_keystroke_boundaries(keystrokes):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(signal, sr=samplerate, color='#1f77b4', axis="s", label='Waveform')
    keystrokes = [(start / 44100, end / 44100) for start, end in keystrokes]
    for i, (start, end) in enumerate(keystrokes):
        if i == 0:
            plt.axvline(x=start, color='g', linestyle='-', linewidth=0.5, label='Start')
            plt.axvline(x=end, color='m', linestyle='-', linewidth=0.5, label='End')
        else:
            plt.axvline(x=start, color='g', linestyle='-', linewidth=0.5)
            plt.axvline(x=end, color='m', linestyle='-', linewidth=0.5)
    plt.legend()
    plt.show()

def isolate_keystrokes(keystroke_boundaries, signal):
    keystrokes = []
    for start, end in keystroke_boundaries:
        keystrokes.append(signal[start:end])
    return keystrokes

def plot_extracted_keystrokes(extracted_keystrokes, samplerate):
    fig, axs = plt.subplots(5, 5, figsize=(15, 20), constrained_layout=True)

    for i, keystroke in enumerate(extracted_keystrokes):
        row = i // 5
        col = i % 5
        librosa.display.waveshow(keystroke, sr=samplerate, color='#1f77b4', ax=axs[row, col], max_points=1000)
        axs[row, col].xaxis.set_major_locator(plt.MaxNLocator(5))
        axs[row, col].set_title(f'Keystroke {i+1}')
        axs[row, col].set_xlabel("")
        axs[row, col].set_ylabel("")

    fig.suptitle('Extracted Keystrokes')
    fig.supxlabel('Time (s)')
    fig.supylabel('Amplitude')
    plt.show()

if __name__ == '__main__':
    window_size = 1024
    hop_size = 225
    threshold = 0.1
    file_path = 'Recordings\A.wav'

    signal, samplerate = load_recording(file_path)
    plot_waveform(signal, samplerate)
    energy = process_keystrokes(signal, window_size, hop_size)
    plot_energy(energy, samplerate, threshold, window_size, hop_size)
    peaks = isolate_keystroke_peaks(energy)
    plot_peaks(peaks, energy)
    keystroke_boundaries = find_keystroke_boundaries(peaks, signal, len(energy), 2851, 11549)
    plot_keystroke_boundaries(keystroke_boundaries)
    extracted_keystrokes = isolate_keystrokes(keystroke_boundaries, signal)
    plot_extracted_keystrokes(extracted_keystrokes, samplerate)