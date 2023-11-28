import torch
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
scriptpath = "."
sys.path.append(os.path.abspath(scriptpath))
from src.models.utils import train_model, get_device, get_dataset, set_seed

class ComplexLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(ComplexLSTM, self).__init__()
        hidden_size = 16
        num_layers = 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size * 2, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out, _ = self.lstm2(out)

        out = self.linear(out[:, -1, :])

        return out


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

    model = ComplexLSTM(input_size=225, output_size=len(idx_to_class))
    criterion = nn.CrossEntropyLoss()

    train_model(model, 300, criterion, train_dataloader, test_dataloader, save_ckpt_path='models/complex_lstm/best.pt', device=device)

    checkpoint = torch.load('models/complex_lstm/best.pt')
    
    print(f"Best F1 score: {max(checkpoint['val_scores'])}")