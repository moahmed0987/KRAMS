import torch
import os
import model_evaluator as me
import trained_model_data_loader as tmdl
from RecordingDataset import RecordingDataset

BASE_DIR = os.path.join("drive", "MyDrive", "Results", "20240427130931")
DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(BASE_DIR, "Model")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "Checkpoints")

dataset = tmdl.get_dataset(DATA_DIR)
test_dataset = tmdl.get_test_dataset(DATA_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = me.load_and_prepare_model(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_130.pth"), device)

print_length = 520

mel_spectrograms = [mel_spectrogram for mel_spectrogram, _ in test_dataset]
output_text = me.predict_keystrokes(model, mel_spectrograms[:print_length], device)

print(output_text)


sentence = [dataset.get_label_from_target(target.item()) for _,target in test_dataset]
sentence = "".join(sentence[:print_length])
print(sentence)


me.check_accuracy(sentence.upper(), output_text.upper())