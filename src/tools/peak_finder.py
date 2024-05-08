import os
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.fft import fft

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import src.keystroke_extractor as ke


def isolate_keystroke_peaks(energy, num_peaks=25):
    for i in [x / 100.0 for x in range(1, 101, 1)]:
        peaks, _ = scipy.signal.find_peaks(energy, prominence=i)
        if len(peaks) == num_peaks:
            break
    return peaks

min_diff = np.inf
for i in range(0, 26):
    FILE_PATH = os.path.join("PCRecordings", chr(65+i)+".wav")
    WINDOW_SIZE = 1023
    HOP_SIZE = 225
    signal, samplerate = ke.load_recording(FILE_PATH)
    energy = ke.process_keystrokes(signal, WINDOW_SIZE, HOP_SIZE)
    
    print(FILE_PATH)
    peaks = isolate_keystroke_peaks(energy)
    if len(peaks) != 25:
        print("Peaks not found.")
        print()

    if len(peaks) != 0:
        ke.plot_peaks(peaks, energy, signal, WINDOW_SIZE, HOP_SIZE, samplerate)
        print("np.diff(peaks):", np.diff(peaks))
        print("min(np.diff(peaks)):", min(np.diff(peaks)))
        if input("Correctly identified peaks? (y/n): ") == "n":
            print()
            continue
        if min(np.diff(peaks)) < min_diff:
            min_diff = min(np.diff(peaks))
        print()
print("Minimum difference between peaks:", min_diff)