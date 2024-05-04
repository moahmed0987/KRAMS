import os

import numpy as np
import torch
from torch.utils.data import Subset

from coatnet import CoAtNet
from RecordingDataset import RecordingDataset


def load_and_prepare_model(model_path, device):
    num_blocks = [2, 2, 12, 28, 2]
    channels = [192, 192, 384, 768, 1536]
    model = CoAtNet((64, 64), 1, num_blocks, channels, num_classes=26)

    if model_path.endswith("model.pth"):
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    else:
        model_checkpoint = torch.load(model_path, map_location=torch.device(device))
        model_state = model_checkpoint['model_state_dict']
        model.load_state_dict(model_state)

    model.to(device)
    model.eval()
    return model

def get_dataset(DATA_DIR):
    dataset = torch.load(os.path.join(DATA_DIR, "dataset.pth"))
    return dataset

def get_train_dataset(DATA_DIR):
    dataset = get_dataset(DATA_DIR)
    train_indices = np.load(os.path.join(DATA_DIR, "train_indices.npy"))
    train_dataset = Subset(dataset, train_indices)
    return train_dataset

def get_validation_dataset(DATA_DIR):
    dataset = get_dataset(DATA_DIR)
    validation_indices = np.load(os.path.join(DATA_DIR, "validation_indices.npy"))
    validation_dataset = Subset(dataset, validation_indices)
    return validation_dataset

def get_test_dataset(DATA_DIR):
    dataset = get_dataset(DATA_DIR)
    test_indices = np.load(os.path.join(DATA_DIR, "test_indices.npy"))
    test_dataset = Subset(dataset, test_indices)
    return test_dataset

if __name__ == "__main__":
    DATA_DIR = os.path.join("Results", "20240427130931", "Data")
    train_dataset = get_train_dataset(DATA_DIR)
    validation_dataset = get_validation_dataset(DATA_DIR)
    test_dataset = get_test_dataset(DATA_DIR)
    print("len(train_dataset): ", len(train_dataset))
    print("len(validation_dataset): ", len(validation_dataset))
    print("len(test_dataset): ", len(test_dataset))
    print("test_dataset[0][0].shape: ", test_dataset[0][0].shape)
