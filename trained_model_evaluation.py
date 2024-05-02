import torch
import os
import model_evaluator as me
import trained_model_data_loader as tmdl
from RecordingDataset import RecordingDataset
from sklearn import metrics
import matplotlib.pyplot as plt

BASE_DIR = os.path.join("drive", "MyDrive", "Results", "20240427130931")
DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(BASE_DIR, "Model")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "Checkpoints")

dataset = tmdl.get_dataset(DATA_DIR)
test_dataset = tmdl.get_test_dataset(DATA_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = me.load_and_prepare_model(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_130.pth"), device)

print_length = len(test_dataset)

mel_spectrograms = [mel_spectrogram for mel_spectrogram, _ in test_dataset]
output_labels = me.predict_keystrokes(model, mel_spectrograms[:print_length], device)
print(output_labels)

target_labels = [dataset.get_label_from_target(target.item()) for _, target in test_dataset]
target_labels = target_labels[:print_length]
print(target_labels)

me.check_accuracy([letter.upper() for letter in target_labels], [letter.upper() for letter in output_labels])

print(metrics.classification_report(target_labels, output_labels))
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
cmd = metrics.ConfusionMatrixDisplay.from_predictions(target_labels, output_labels, ax=ax)
plt.show()