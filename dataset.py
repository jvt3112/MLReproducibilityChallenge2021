import numpy as np
import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def createDataset(userOpt, train):
    """ Prepares the Dataset

    Args:
        userOpt: all the parsing options provided by user/default
        train: bool param: bool to check for preparing dataset for train or test 
    """
    transform = T.Compose([
            T.ToTensor(), 
            T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, 
                        np.array([63.0, 62.1, 66.7]) / 255.0),])
    if train:
        transform = T.Compose([
            T.Pad(4, padding_mode='reflect'), 
            T.RandomHorizontalFlip(), 
            T.RandomCrop(32), 
            transform])
    return getattr(datasets, 'CIFAR10')('.', train=train, download=True, transform=transform)

def createTrainLoader(userOpt, mode):
    """ creates a Train Loader

    Args:
        userOpt: all the parsing options provided by user/default
        mode: mode checks for creating train loader for train or test 
    """
    return DataLoader(createDataset(userOpt, mode), userOpt.batch_size, shuffle=mode, num_workers=1, pin_memory=torch.cuda.is_available())