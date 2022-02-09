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

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    batch_size = 4

    
    def __init__(self, dName=None, dDescription=None):    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    #
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        super().__init__(dName, dDescription)
    
    def load(self):
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
            X_test.append(pair['image'])
            y_test.append(pair['label'])
        print('training set size:', len(data['train']), 'testing set size:', len(data['test']))

        return {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}