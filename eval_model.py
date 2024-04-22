import os

import torch

import feature_extractor as fe
import keystroke_extractor as ke
from coatnet import CoAtNet


def load_and_prepare_model(model_path, device):
    num_blocks = [2, 2, 12, 28, 2]
    channels = [192, 192, 384, 768, 1536]
    model = CoAtNet((64, 64), 1, num_blocks, channels, num_classes=26)

    if model_path.endswith("model.pth"):
        model.load_state_dict(torch.load(model_path))
    else:
        try:
            model_checkpoint = torch.load(model_path)
        except RuntimeError:
            model_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        model_state = model_checkpoint['model_state_dict']
        model.load_state_dict(model_state)

    model.to(device)
    model.eval()
    return model

def extract_keystrokes_and_features(sentence, recording_path, WINDOW_SIZE, HOP_SIZE, BEFORE, AFTER):
    signal, samplerate = ke.load_recording(recording_path)
    energy = ke.process_keystrokes(signal, WINDOW_SIZE, HOP_SIZE)
    peaks = ke.isolate_keystroke_peaks(energy)
    keystroke_boundaries = ke.find_keystroke_boundaries(peaks, signal, len(energy), BEFORE, AFTER)
    extracted_keystrokes = ke.isolate_keystrokes(keystroke_boundaries, signal)
    mel_spectrograms = [fe.generate_mel_spectrogram(keystroke, samplerate, WINDOW_SIZE, HOP_SIZE) for keystroke in extracted_keystrokes]
    df_relative_paths, df_labels, df_targets, df_mel_spectrograms = [],[],[],[]
    for i, mel_spectrogram in enumerate(mel_spectrograms):
        df_relative_paths.append(recording_path)
        df_labels.append(sentence[i])
        df_targets.append(ord(sentence[i])-65)
        df_mel_spectrograms.append(torch.tensor(mel_spectrogram))
    return df_relative_paths, df_labels, df_targets, df_mel_spectrograms

def predict_keystrokes(model, df_mel_spectrograms, device):
    output_text = []
    with torch.no_grad():
        for ms in df_mel_spectrograms:
            ms = ms.to(device)
            ms = torch.unsqueeze(ms, 0)
            ms = torch.unsqueeze(ms, 0)
            output = model(ms)
            pred = output.argmax(dim=1, keepdim=True)
            output_text.append(chr(pred.item()+65))
            print(chr(pred.item()+65), end=" ")
        print()
    return "".join(output_text)


def check_accuracy(sample_text, output_text):
    correct = 0
    for i in range(len(sample_text)):
        if output_text[i] == sample_text[i]:
            correct += 1
    print("Accuracy:", correct / len(sample_text), "%")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentence = "ThisIsATestSentenceSample"
    recording_path = os.path.join("drive", "MyDrive", "Recordings", "ThisIsATestSentenceSample.wav")
    model_path = os.path.join("drive", "MyDrive", "Results", "20240321182511", "Checkpoints", "checkpoint_epoch_610.pth")
    BEFORE = int(0.3 * 14400)
    AFTER = int(0.7 * 14400)
    WINDOW_SIZE = 1023
    HOP_SIZE = 225
    model = load_and_prepare_model(model_path, device)
    df_relative_paths, df_labels, df_targets, df_mel_spectrograms = extract_keystrokes_and_features(sentence, recording_path, WINDOW_SIZE, HOP_SIZE, BEFORE, AFTER)
    output_text = predict_keystrokes(model, df_mel_spectrograms, device)
    check_accuracy(sentence.upper(), output_text.upper())
