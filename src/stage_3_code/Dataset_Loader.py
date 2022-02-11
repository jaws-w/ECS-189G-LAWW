'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from src.base_class.dataset import dataset
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
from torch.utils.data import Dataset, DataLoader
import math

class CustomDataset(Dataset):
    def __init__(self, img, labels):
        self.labels = labels
        self.images = img

    def __len__(self):
            return len(self.labels)

    def __getitem__(self, idx):
            label = self.labels[idx]
            image = self.images[idx]
            sample = image, label
            return sample

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )

    
    def __init__(self, dName=None, dDescription=None):    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    #
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        super().__init__(dName, dDescription)
    
    def load(self):
        batch_size = 100

        print('loading data...')
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb') # or change MNIST to other dataset names (CIFAR, ORL)
        data = pickle.load(f)
        f.close()
        for pair in data['train']:
            X_train.append(self.transform(pair['image']))
            y_train.append(pair['label'])
        for pair in data['test']:
            X_test.append(self.transform(pair['image']))
            y_test.append(pair['label'])
        print('training set size:', len(data['train']), 'testing set size:', len(data['test']))

        trainset_data = CustomDataset(X_train, y_train)
        testset_data = CustomDataset(X_test, y_test)
        #return {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        # return {'train': trainset_data, 'test': testset_data}

        trainloader = DataLoader(trainset_data, batch_size=batch_size, shuffle=True, num_workers=0)
        testloader = DataLoader(testset_data, batch_size=batch_size, shuffle=False, num_workers=0)

        return {'train': trainloader, 'test': testloader}