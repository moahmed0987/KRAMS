import torch
import numpy as np
from torch.utils.data import Subset
import os
from RecordingDataset import RecordingDataset


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
