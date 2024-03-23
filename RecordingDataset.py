import numpy as np
import pandas as pd
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
    
    def get_label(self, idx):
        return self.df.iloc[idx, 2]
    
    def get_label_from_target(self, target):
        return self.df[self.df['target'] == target].iloc[0, 2]
    
    # RecordingDataset must be initialised with the dataframe that created the
    # mel_spectrogram due to the randomness of data augmentation
    def get_id_from_mel_spectrogram(self, mel_spectrogram):
        for i in range(len(self.df)):
            if np.array_equal(self.df['mel_spectrogram'].iloc[i], mel_spectrogram):
                break
        return self.df.iloc[i]['id']
    

if __name__ == "__main__":
    mel_spec = np.load("mel_spectrogram.npy")
    import training_data_processor as tdp
    RECORDINGS_DIR = "Recordings"
    WINDOW_SIZE = 1023
    HOP_SIZE = 225
    BEFORE = int(0.2 * 14400)
    AFTER = int(0.8 * 14400)

    file_paths = tdp.get_file_paths(RECORDINGS_DIR)
    df_relative_paths, df_labels, df_targets, df_mel_spectrograms = tdp.process_recordings(file_paths, WINDOW_SIZE, HOP_SIZE, BEFORE, AFTER)
    df = tdp.to_dataframe(df_relative_paths, df_labels, df_targets, df_mel_spectrograms)
    dataset = RecordingDataset(df, "Recordings")
    np.save("dataset.npy", dataset.df)

    ds = np.load("dataset.npy", allow_pickle=True)
    ds = pd.DataFrame(ds, columns=["id", "relative_path", "label", "target", "mel_spectrogram"])
    id = RecordingDataset(ds, "Recordings").get_id_from_mel_spectrogram(mel_spec)
    print(id)
