import torch

import sys
import os

scriptpath = "."
sys.path.append(os.path.abspath(scriptpath))
from src.models.utils import get_device, get_dataset
from src.models.simple_lstm.train_model import SimpleLSTM

if __name__ == '__main__':
    device = get_device()
    print('Connected to device:', device)
    train_dataset, test_dataset, idx_to_class = get_dataset()
    model = SimpleLSTM(input_size=225, output_size=len(idx_to_class))
    
    checkpoint = torch.load('models/simple_lstm/best.pt')
    
    model.load_state_dict(checkpoint['model_state_dict'])
