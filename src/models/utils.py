from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

import cv2

import os
from os.path import exists
import torch.nn.functional as F

import random
import numpy as np

import sys
scriptpath = "."
sys.path.append(os.path.abspath(scriptpath))
from src.utils import create_dataframe

def set_seed():
    """
    Set random seeds for reproducibility.
    """
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

def read_classes(path):
    classes = dict()
    with open(path, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            key = int(line[0])
            value = line[1]
            classes[key] = value
    return classes

def get_max_frame_count(source_folder : str, dataset : pd.DataFrame) -> int:
    max_frame_count = 0
    for key, row in dataset.iterrows():
        class_name = row['class_name']
        file_name = row['video_name']
        file_path = os.path.join(source_folder, class_name, file_name+'.mp4')
        cap = cv2.VideoCapture(file_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frame_count = max(max_frame_count, length)
    return max_frame_count



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
    
    train_dataset = ASLDataset('data/internal/features', train, max_frame_count, class_to_idx)
    test_dataset = ASLDataset('data/internal/features', test, max_frame_count, class_to_idx)

    return train_dataset, test_dataset, idx_to_class

class ASLDataset(Dataset):
    """
    A custom dataset class for loading American Sign Language (ASL) videos and their corresponding labels.

    Args:
        tensors_folder (str): Path to the folder containing the .pt arrays (converted videos).
        dataset_df (DataFrame): Pandas dataframe with information about videos
        max_frame_count (int): Maximum length of video in frames.
        ohe (OneHotEncoder): Required to encode class id to one hot numeric array.

    Attributes:
        tensors_folder (str): Path to the folder containing the .pt arrays (converted videos).
        name_with_label (dict): A dictionary mapping video names to their labels.
        classes (list): A list of class names.
    """

    def __init__(self, tensors_folder: str, dataset_df: pd.DataFrame, max_frame_count: int, class_to_idx) -> None:
        # Default frame (set of points) to insert in short videos to make all videos equal in length
        default_frame = [0] * (21 * 3 * 2 + 33 * 3)
        
        self.tensor_sequences = []
        self.targets = []
        for i, row in dataset_df.iterrows():
            path_to_tensor = tensors_folder + '/' + row["video_name"] + '.pt'
            
            if not exists(path_to_tensor):
                continue
            
            # Open the video file using OpenCV
            tensor = torch.load(path_to_tensor)

            # Change length of video (not a video but sequence of frames with points coordinates in each frame)
            # pad_right = [default_frame for _ in range(0, max_frame_count - tensor.shape[0])]
            # pad_right = torch.tensor(pad_right)
            # self.tensor_sequences.append(torch.cat((pad_right, F.normalize(tensor.float(), dim=1))))
            self.tensor_sequences.append(F.normalize(tensor.float(), dim=1))
            self.targets.append(class_to_idx[dataset_df["class_id"][i]])
        self.targets = torch.tensor(self.targets)
            
            
    def __len__(self):
        """
        Returns the number of videos in the dataset.

        Returns:
            int: The number of videos in the dataset.
        """
        return len(self.targets)

    def __getitem__(self, index):
        """
        Retrieves a video and its corresponding label from the dataset.

        Args:
            index (int): The index of the video to retrieve.

        Returns:
            tuple: A tuple containing the video frames as a PyTorch tensor and the label.
        """

        return self.tensor_sequences[index].float(), self.targets[index].long()

def train_model(model: nn.Module, epochs: int, criterion, train_dataloader, validation_dataloader, load_ckpt : bool =False, load_ckpt_path : str or None = None, save_ckpt_path : str='models/best.pt', device : torch.device='cuda'):
    """
    Function that trains model using number of epochs, loss function, optimizer.
    Can use validation or test data set for evaluation.
    Calculates f1 score.

    Parameter
    ---------
    model : nn.Module
      Model to train.
    epochs: int
      Number of train epochs
    criterion
      The loss function from pytorch
    optimizer
      The optimizer from pytorch
    """
    
    if load_ckpt_path is None:
        load_ckpt_path = save_ckpt_path

    model.train()
    model.to(device)
    
    # best score for checkpointing
    best = 0.0
    train_losses = []
    train_scores = []
    val_losses = []
    val_scores = []
    
    first_epoch = 1
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    isCkptExists = os.path.isfile(load_ckpt_path)
    
    if (load_ckpt and not isCkptExists):
        print('Checkpoint file does not exist. Training model from scratch!')

    if (load_ckpt and isCkptExists):
        checkpoint = torch.load(load_ckpt_path)
        best = checkpoint['best_score']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint['train_losses']
        train_scores = checkpoint['train_scores']
        val_scores = checkpoint['val_scores']
        val_losses = checkpoint['val_losses']
        first_epoch = checkpoint['epoch'] + 1

    # Train the model
    for epoch in range(first_epoch, epochs + first_epoch):
        model.train()

        predicted_train = []
        true_train = []

        train_loss = 0.0

        bar = tqdm(train_dataloader)
        iterations = 0

        for inputs, targets in bar:
          
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Get predicted classes and true classes from data
            predictions = torch.argmax(outputs, dim=1)
            for item in predictions:
                predicted_train.append(item.cpu().numpy())
            for item in targets:
                true_train.append(item.cpu().numpy())
            iterations += 1
            bar.set_postfix(
                ({"loss": f"{train_loss/(iterations*train_dataloader.batch_size)}"}))

        # Computing loss
        train_loss /= len(train_dataloader.dataset)
        # Computing f1 score
        train_f1 = f1_score(true_train, predicted_train, average="macro")

        # Printing information in the end of train loop
        val_loss, val_f1 = test_model(model, criterion, validation_dataloader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_scores.append(train_f1)
        val_scores.append(val_f1)
                
        if val_f1 > best:
            best = val_f1
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_scores': train_scores,
            'val_losses': val_losses,
            'val_scores': val_scores,
            'best_score': best,
            }, save_ckpt_path)

        print(f"Epoch {epoch+1}: \ntrain:\t\t(loss: {train_loss:.4f}, f1 score: {train_f1:.4f}) \nvalidation:\t(loss: {val_loss:.4f}, f1 score: {val_f1:.4f})\n")


def test_model(model: nn.Module, criterion, test_dataloader: DataLoader, device='cuda'):
    """
    Function that evaluates model on specified dataloader
    by specified loss function.

    Parameter
    ---------
    model : nn.Module
      Model to train.
    criterion
      The loss function from pytorch
    test_dataloader: DataLoader
      The dataset for testing model

    Returns
    -------
    float: loss of model on given dataset
    float: f1 score of model on given dataset
    """

    model.eval()
    model.to(device)

    # Test loss value
    test_loss = 0.0

    # Lists for calculation f1 score
    predicted_test = []
    true_test = []

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            test_loss += criterion(outputs, targets)

            # Get predicted classes and true classes from data
            predictions = torch.argmax(outputs, dim=1)
            for item in predictions:
                predicted_test.append(item.cpu().numpy())
            for item in targets:
                true_test.append(item.cpu().numpy())

    # Computation of test loss
    test_loss /= len(test_dataloader)

    # Computation of f1 score
    test_f1 = f1_score(true_test, predicted_test, average="macro")
    return test_loss.item(), test_f1