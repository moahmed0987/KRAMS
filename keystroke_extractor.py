import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft


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
def plot_energy(energy, samplerate, threshold,window_size, hop_size):
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
    # extract keystrokes from the signal
    for start, end in keystrokes:
        start_sample = int((start / windows) * len(signal))
        end_sample = int((end / windows) * len(signal))
        # add 8820 samples (200ms) to the start and end to make sure the keystroke is fully captured
        start_sample = start_sample - 8820 if start_sample - 8820 > 0 else 0
        end_sample = end_sample + 8820 if end_sample + 8820 < len(signal) else len(signal)
        extracted_keystrokes.append(signal[start_sample:end_sample])

    # remove all silence from the start of each keystroke, and add 10ms of silence to the start
    for i in range(len(extracted_keystrokes)):
        non_silent_indices = np.where(np.abs(extracted_keystrokes[i]) > 0.01)[0]
        start_index = non_silent_indices[0] if non_silent_indices.size else 0
        if start_index > 440:
            start_index -= 441 # 10ms (0.01s * 44100 = 441) before the first non-silent index
        extracted_keystrokes[i] = extracted_keystrokes[i][start_index:]

    # remove silence from the end of each keystroke, and add 10ms of silence to the end
    for i in range(len(extracted_keystrokes)):
        non_silent_indices = np.where(np.abs(extracted_keystrokes[i]) > 0.01)[0]
        end_index = non_silent_indices[-1] if non_silent_indices.size else len(extracted_keystrokes[i])
        if len(extracted_keystrokes[i]) - end_index > 440:
            end_index += 441 # 10ms (0.01s * 44100 = 441) after the last non-silent index 
        extracted_keystrokes[i] = extracted_keystrokes[i][:end_index]

    return extracted_keystrokes

def plot_extracted_keystrokes(extracted_keystrokes, samplerate):
    fig, axs = plt.subplots(5, 5, figsize=(15, 20), constrained_layout=True)

    for i, keystroke in enumerate(extracted_keystrokes):
        row = i // 5
        col = i % 5
        librosa.display.waveshow(keystroke, sr=samplerate, color='#1f77b4', ax=axs[row, col], max_points=1000)
        axs[row, col].set_title(f'Keystroke {i+1}')
        axs[row, col].set_xlabel("")
        axs[row, col].set_ylabel("")

    fig.suptitle('Extracted Keystrokes')
    fig.supxlabel('Time (s)')
    fig.supylabel('Amplitude')
    plt.show()

if __name__ == '__main__':
    window_size = 10000
    hop_size = window_size // 2
    threshold = 0.1
    file_path = 'Recordings\A.wav'

    signal, samplerate = load_recording(file_path)
    plot_waveform(signal, samplerate)
    energy = process_keystrokes(signal, window_size, hop_size)
    plot_energy(energy, samplerate, threshold, window_size, hop_size)
    keystrokes = isolate_keystrokes(energy, threshold)
    plot_keystrokes(keystrokes)
    extracted_keystrokes = extract_keystrokes(keystrokes, signal, len(energy))
    plot_extracted_keystrokes(extracted_keystrokes, samplerate)