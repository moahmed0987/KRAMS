import torch
from torch.utils.data import Dataset

class RecordingDataset(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = data_path

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # return 'mel_spectrogram' and 'target' from idx-th row of the dataframe as tensors
        mel_spectrogram = self.df.iloc[idx, 4]
        target = self.df.iloc[idx, 3]
        tensor_mel_spectrogram = torch.tensor(mel_spectrogram)
        tensor_target = torch.tensor(target)
        return tensor_mel_spectrogram, tensor_target
        
