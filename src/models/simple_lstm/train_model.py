import torch
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
scriptpath = "."
sys.path.append(os.path.abspath(scriptpath))
from src.models.utils import train_model, get_device, get_dataset, set_seed

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


if __name__ == '__main__':
    set_seed()
    
    device = get_device()
    print('Connected to device:', device)
    
    # Create datasets
    train_dataset, test_dataset, idx_to_class = get_dataset()
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Create dataloaders
    batch_size = 1

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleLSTM(input_size=225, output_size=len(idx_to_class))
    criterion = nn.CrossEntropyLoss()

    train_model(model, 200, criterion, train_dataloader, test_dataloader, save_ckpt_path='models/simple_lstm/best.pt', device=device)

    checkpoint = torch.load('models/simple_lstm/best.pt')
    
    print(f"Best F1 score: {max(checkpoint['val_scores'])}")