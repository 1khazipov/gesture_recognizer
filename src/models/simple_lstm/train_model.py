import torch
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

scriptpath = "."
sys.path.append(os.path.abspath(scriptpath))
from src.utils import create_dataframe
from src.models.utils import train_model, ASLDataset, get_max_frame_count, read_classes

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.lstm = nn.Sequential(
            nn.LSTM(input_size, 64, 1, batch_first=True, bidirectional=True),
        )
        self.linear = nn.Sequential(
            nn.Linear(64 * 2, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:,-1,:])

        return x

def get_device():
    #Set up device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def get_dataset():
    source_folder = os.path.join('data', 'internal', 'preprocessed_videos')

    class_file_name = "data/raw/dataset/wlasl_class_list.txt"

    dataset = create_dataframe(source_folder, class_file_name)

    idx_to_class = [ c_id for i, c_id in enumerate(set(dataset.class_id))]
    class_to_idx = { c_id: i for i, c_id in enumerate(idx_to_class)}

    train, test = train_test_split(dataset, test_size=0.2, random_state=42)

    max_frame_count = get_max_frame_count(source_folder, dataset)
        
    print('Maximum frame count:',max_frame_count)

    train_dataset = ASLDataset('data/internal/features', train, max_frame_count, class_to_idx)
    test_dataset = ASLDataset('data/internal/features', test, max_frame_count, class_to_idx)

    return train_dataset, test_dataset


if __name__ == '__main__':
    device = get_device()
    print('Connected to device:', device)
    
    # Create datasets
    train_dataset, test_dataset = get_dataset()
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Create dataloaders
    batch_size = 1

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleLSTM(input_size=225, output_size=len(set(train_dataset.targets)))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = train_model(model, 200, criterion, optimizer, train_dataloader, test_dataloader, ckpt_path='models/simple_lstm/best.pt', device=device)

    print(f"Best F1 score: {max(history['val_scores'])}")