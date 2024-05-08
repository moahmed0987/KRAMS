import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import src.training_data_processor as tdp

# check mel spectrogram shapes are correct
for i in range(0, 26):
    mel_spectrograms = tdp.data_processing_pipeline("Recordings/" + chr(65 + i) + ".wav", 1023, 225, int(0.2 * 14400), int(0.8 * 14400))
    for i in mel_spectrograms:
        if i.shape != (64, 64):
            print("Anomaly found in mel spectrogram shape:")
            print(i.shape)