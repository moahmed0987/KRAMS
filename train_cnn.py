import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

import RecordingDataset
import training_data_processor as tdp

WINDOW_SIZE = 1023
HOP_SIZE = 225
BEFORE = int(0.2 * 14400)
AFTER = int(0.8 * 14400)
NUM_EPOCHS = 1100
EPOCHS_PER_CHECKPOINT = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
RECORDINGS_DIR = "Recordings"
BASE_DIR = os.path.join("Results", datetime.now().strftime("%Y%m%d%H%M%S"))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "Checkpoints")
MODEL_DIR = os.path.join(BASE_DIR, "Model")
FIGURE_DIR = os.path.join(BASE_DIR, "Figures")

file_paths = tdp.get_file_paths(RECORDINGS_DIR)
df_relative_paths, df_labels, df_targets, df_mel_spectrograms = tdp.process_recordings(file_paths, WINDOW_SIZE, HOP_SIZE, BEFORE, AFTER)
df = tdp.to_dataframe(df_relative_paths, df_labels, df_targets, df_mel_spectrograms)

train_indices = random.sample(range(len(df)), int(0.8 * len(df)))
test_indices = list(set(range(len(df))) - set(train_indices))

train_df = df.iloc[train_indices]
test_df = df.iloc[test_indices]

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * (64//4) * (64//4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * (64//4) * (64//4))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ConvNet(num_classes=26)

optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()
train_dataset = RecordingDataset.RecordingDataset(train_df, "Recordings")
test_dataset = RecordingDataset.RecordingDataset(test_df, "Recordings")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
        if batch_idx % (len(train_loader) // 10) == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss = running_loss / len(train_loader)
    return train_loss
            
def test(model, device, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    incorrect = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = torch.unsqueeze(data, 1)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            misclassified = (pred != target.view_as(pred)).squeeze()
            for idx, is_misclassified in enumerate(misclassified):
                if is_misclassified:
                    incorrect.append((data[idx], target[idx], pred[idx]))

    test_loss = running_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    print("Misclassified samples:")
    for data, target, pred in incorrect:
        # print(f"Sample: {data.cpu().squeeze().numpy()}")
        print(f"Target: {target.item()}, Predicted: {pred.item()}")
        print(f"Target label: {test_dataset.get_label_from_target(target.item())}, Predicted label: {test_dataset.get_label_from_target(pred.item())}")
        print("-" * 20)

    return test_loss

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train(model, device, train_loader, optimiser, criterion, epoch)
        train_losses.append(train_loss)
        if epoch % EPOCHS_PER_CHECKPOINT == 0:
            test_loss = test(model, device, test_loader, criterion)
            test_losses.append(test_loss)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimiser_state_dict': optimiser.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss
                }, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch} with test loss {test_loss:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot([(i+1) * EPOCHS_PER_CHECKPOINT for i in range(len(test_losses))], test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURE_DIR, f"loss.png"))
    print(f"Loss plot saved at {os.path.join(FIGURE_DIR, 'loss.png')}")
    print("train_losses: ", train_losses)
    print("test_losses: ", test_losses)

if __name__ == "__main__":
    run()