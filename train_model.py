import os
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from tqdm import tqdm

import feature_extractor as fe
import keystroke_extractor as ke
import RecordingDataset as RecordingDataset
import training_data_processor as tdp
from coatnet import CoAtNet

def train(model, device, train_loader, optimiser, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = torch.unsqueeze(data, 1)
        optimiser.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()
        if batch_idx % (len(train_loader) // 4) == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss = running_loss / len(train_loader)
    return train_loss

def validation(model, device, validation_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    incorrect = []
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            data = torch.unsqueeze(data, 1)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            misclassified = (pred != target.view_as(pred)).squeeze()
            if misclassified.ndim != 0:
                for idx, is_misclassified in enumerate(misclassified):
                    if is_misclassified:
                        incorrect.append((data[idx], target[idx], pred[idx]))

    validation_loss = running_loss / len(validation_loader)
    accuracy = 100. * correct / len(validation_loader.dataset)
    print("\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        validation_loss, correct, len(validation_loader.dataset), accuracy))

    print("Misclassified samples:")
    for data, target, pred in incorrect:
        # print(f"Sample: {data.cpu().squeeze().numpy()}")
        print(f"Target: {target.item()}, Predicted: {pred.item()}")
        print(f"Target label: {validation_loader.dataset.dataset.get_label_from_target(target.item())}, Predicted label: {validation_loader.dataset.dataset.get_label_from_target(pred.item())}")
        print("-" * 20)

    return validation_loss

def run(RECORDINGS_DIR, WINDOW_SIZE, HOP_SIZE, BEFORE, AFTER, NUM_EPOCHS, EPOCHS_PER_CHECKPOINT, BATCH_SIZE, LEARNING_RATE):
    BASE_DIR = os.path.join("Results", datetime.now().strftime("%Y%m%d%H%M%S"))
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "Checkpoints")
    MODEL_DIR = os.path.join(BASE_DIR, "Model")
    FIGURE_DIR = os.path.join(BASE_DIR, "Figures")
    DATA_DIR = os.path.join(BASE_DIR, "Data")

    file_paths = tdp.get_file_paths(RECORDINGS_DIR)
    df_relative_paths, df_labels, df_targets, df_mel_spectrograms = tdp.process_recordings(file_paths, WINDOW_SIZE, HOP_SIZE, BEFORE, AFTER)
    df = tdp.to_dataframe(df_relative_paths, df_labels, df_targets, df_mel_spectrograms)

    num_blocks = [2, 2, 12, 28, 2]
    channels = [192, 192, 384, 768, 1536]
    model = CoAtNet(image_size=(64, 64), in_channels=1, num_blocks=num_blocks, channels=channels, num_classes=26)

    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    dataset = RecordingDataset(df, RECORDINGS_DIR)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    validation_losses = []
    best_validation_loss = float('inf')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
        train_loss = train(model, device, train_loader, optimiser, criterion, epoch)
        train_losses.append(train_loss)
        if epoch % EPOCHS_PER_CHECKPOINT == 0:
            validation_loss = validation(model, device, validation_loader, criterion)
            validation_losses.append(validation_loss)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimiser_state_dict': optimiser.state_dict(),
                    'train_loss': train_loss,
                    'validation_loss': validation_loss
                }, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch} with validation loss {validation_loss:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot([(i+1) * EPOCHS_PER_CHECKPOINT for i in range(len(validation_losses))], validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURE_DIR, f"loss.png"))
    print(f"Loss plot saved at {os.path.join(FIGURE_DIR, 'loss.png')}")
    print("train_losses: ", train_losses)
    print("validation_losses: ", validation_losses)

    os.makedirs(DATA_DIR, exist_ok=True)
    torch.save(dataset, os.path.join(DATA_DIR, "dataset.pth"))
    np.save(os.path.join(DATA_DIR, "train_indices.npy"), train_dataset.indices)
    np.save(os.path.join(DATA_DIR, "validation_indices.npy"), validation_dataset.indices)
    np.save(os.path.join(DATA_DIR, "test_indices.npy"), test_dataset.indices)

    return BASE_DIR

if __name__ == "__main__":
    WINDOW_SIZE = 1023
    HOP_SIZE = 225
    BEFORE = int(0.3 * 14400)
    AFTER = int(0.7 * 14400)
    NUM_EPOCHS = 1100
    EPOCHS_PER_CHECKPOINT = 10
    BATCH_SIZE = 130
    LEARNING_RATE = 0.0005
    RECORDINGS_DIR = os.path.join("Recordings")
    run(RECORDINGS_DIR, WINDOW_SIZE, HOP_SIZE, BEFORE, AFTER, NUM_EPOCHS, EPOCHS_PER_CHECKPOINT, BATCH_SIZE, LEARNING_RATE)