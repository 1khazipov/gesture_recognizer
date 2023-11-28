import torch
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import optim
import torchvision
from torch import nn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import trange
from torch.utils.data import random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torchmetrics import F1Score


device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16

transform = {
    'train': A.Compose([
    A.ToFloat(max_value=255.0),
    A.RandomCrop(width=320, height=320),
    A.ShiftScaleRotate(),
    #A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Blur(p=0.5),
    A.ImageCompression(p=0.5),
    ToTensorV2() 
]),
    'test': A.Compose([
    A.ToFloat(max_value=255.0),
    ToTensorV2() 
])
}

class MyDataset(torch.utils.data.Dataset):
    """
    Custom dataset for gesture recognition

    """
    def __init__(self, subset, transform=None):
        """

        Args:
            subset (_type_): part of the initial dataset
            transform (_type_, optional): transforms that will be applied. Defaults to None.
        """
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        """

        Args:
            index (_type_): index of the dataset item that will be returned

        Returns frame and label
        """
        x, y = self.subset[index]
        image_np = np.array(x)
        if self.transform:
            x = self.transform(image=image_np)["image"]
        return x, y
        
    def __len__(self):
        """

        Returns len of the dataset
        """
        return len(self.subset)
    
     

img_dataset = torchvision.datasets.ImageFolder(
    root='data/raw/mobile_net_dataset/train_images',
    #transform = Transform(transform)
)

# Let's divide dataset into train and test part
size = len(img_dataset)
train_size = int(size * 0.8)
val_size = size - train_size
train_dataset, valid_dataset = random_split(img_dataset, (train_size, val_size))

train_dataset = MyDataset(train_dataset, transform=transform['train'])
valid_dataset = MyDataset(valid_dataset, transform=transform['test'])

num_classes = len(img_dataset.classes)

# Creating dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=True
)

# Let's initialize model and change the last layer
model = models.mobilenet_v3_small(pretrained=True, progress=True)
model.classifier[-1] = nn.Linear(1024, num_classes)
model.to(device)

# Let's initialize multiclass f1 score
f1_score = F1Score(task="multiclass", num_classes=4).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# This code was inspired by https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train_one_epoch(epoch_index):
    """_summary_

    Args:
        epoch_index (_type_): _description_
        Epoch index used in writer

    Returns:
        _type_: average loss of the batch
    """
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.float().to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 0:
            last_loss = running_loss / 1000 # loss per batch
            print('batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

EPOCHS = 10

best_vloss = 1_000_000.

for epoch in trange(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()
    correct = 0
    f1 = 0
    
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            vinputs = vinputs.float().to(device)
            vlabels = vlabels.to(device)
            
            voutputs = model(vinputs)
            preds = torch.argmax(voutputs, 1)
            f1 += f1_score(preds, vlabels)
            correct += (vlabels == preds).sum()
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    acc = correct / len(valid_dataset)
    print(f"Accuracy: {acc}")
    print(f"Average f1: {f1 / (i+1)}")
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'models/mobile_net/model{}_{}.pth'.format(timestamp, epoch)
        torch.save(model.state_dict(), model_path)


inputs, labels = next(iter(train_loader))

inputs = inputs.to(device)
labels = labels.to(device)
out = model(inputs)
labels = torch.argmax(out, 1)
print(labels)
