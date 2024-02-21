import matplotlib.pyplot as plt
import numpy as np
import keystroke_extractor as ke
import librosa

def mel_spectrogram(signal, samplerate, window_size, hop_size):
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=samplerate, n_mels=64, win_length=window_size, hop_length=hop_size)
    return mel_spectrogram

def mel_spectrograms(signals, samplerate):
    return [mel_spectrogram(signal, samplerate) for signal in signals]

def display_mel_spectrograms(mel_spectrograms, samplerate, window_size, hop_size):
    fig, axs = plt.subplots(5, 5, figsize=(15, 15), constrained_layout=True)
    for i, mel_spectrogram in enumerate(mel_spectrograms):
        row = i // 5
        col = i % 5
        mel_show = librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', x_axis='time', ax=axs[row, col], win_length=window_size, hop_length=hop_size, sr=samplerate)
        axs[row, col].set_title('Mel spectrogram ' + str(i+1))
        axs[row, col].xaxis.set_major_locator(plt.MaxNLocator(5))
        axs[row, col].set_xlabel("")
        axs[row, col].set_ylabel("")

    fig.colorbar(mel_show, ax=axs.ravel().tolist(), format='%+2.0f dB')
    fig.supxlabel('Time (s)')
    fig.supylabel('Mels (Hz)')
    plt.show()

if __name__ == "__main__":
    window_size = 1024
    hop_size = 225
    threshold = 0.1
    file_path = 'Recordings\A.wav'
    signal, samplerate = ke.load_recording(file_path)
    energy = ke.process_keystrokes(signal, window_size, hop_size)
    peaks = ke.isolate_keystroke_peaks(energy)
    keystroke_boundaries = ke.find_keystroke_boundaries(peaks, signal, len(energy), 2851, 11549)
    extracted_keystrokes = ke.isolate_keystrokes(keystroke_boundaries, signal)
    print(f"Extracted {len(extracted_keystrokes)} keystrokes")
    mel_spectrograms = [mel_spectrogram(keystroke, samplerate, window_size, hop_size) for keystroke in extracted_keystrokes]
    print(f"Computed {len(mel_spectrograms)} mel spectrograms")
    print(mel_spectrograms)
    ke.plot_extracted_keystrokes(extracted_keystrokes, samplerate)
    display_mel_spectrograms(mel_spectrograms, samplerate, window_size, hop_size)