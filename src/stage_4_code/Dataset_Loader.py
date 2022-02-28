'''
Concrete IO class for a specific dataset
'''
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from src.base_class.dataset import dataset
from src.base_class.preprocess_helpers import ConstructVocab, set_tensor_padding, set_padding

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

# Gets data from csv files that contain all the reviews data in a single file.
class Classification_Dataset(Dataset):

    def __init__(self, file_name, transform=None, target_transform=None) -> None:
        super().__init__()
        self.data = pd.read_csv(file_name)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data['review'][index], self.data['rating'][index]


class Model_Dataset(Dataset):
    def __init__(self, inputs, labels):
        self.data = inputs
        self.target = labels

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        return x, y#, x_len

    def __len__(self):
        return len(self.data)


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    label_size = None
    vocab_size = None
    vocab = None

    def __init__(self, dName=None, dDescription=None, batch_size=1):
        super().__init__(dName, dDescription)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size
    
    def load(self):

        print('loading data...')
        
        if self.dataset_name == 'CLASSIFICATION':
            train_data = Classification_Dataset(self.dataset_source_folder_path + '/train.csv')
            test_data = Classification_Dataset(self.dataset_source_folder_path + '/test.csv')

            all_data = pd.concat([train_data.data['review'], test_data.data['review']]).to_list()
            self.vocab = ConstructVocab(all_data)
            self.vocab_size = len(self.vocab.word_to_idx)

            self.label_size = train_data.data['rating'][0]

            # Create input tensors.
            train_tensor = [[self.vocab.word_to_idx[word] for word in review.split()] for review in train_data.data['review']]
            test_tensor = [[self.vocab.word_to_idx[word] for word in review.split()] for review in test_data.data['review']]

            # Add padding.
            max_len_train_input = max(len(i) for i in train_tensor)
            train_tensor = torch.LongTensor(set_tensor_padding(train_tensor, max_len_train_input)).to(self.device)

            max_len_test_input = max(len(i) for i in test_tensor)
            test_tensor = torch.LongTensor(set_tensor_padding(test_tensor, max_len_test_input)).to(self.device)

            # Convert the processed data into DataLoader class so that Method_RNN can use it easily.
            train_dataset = Model_Dataset(train_tensor, torch.LongTensor(train_data.data['rating']).to(self.device))
            test_dataset = Model_Dataset(test_tensor, torch.LongTensor(test_data.data['rating']).to(self.device))

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


            # print(train_inputs.idx_to_word[20])

            return {'train': train_loader, 'test': test_loader}

        elif self.dataset_name == 'GENERATION':
            seq_len = 3
            self.label_size = seq_len

            data = Classification_Dataset(self.dataset_source_folder_path + '/cleaned.csv')
            self.vocab = ConstructVocab(data.data['joke'])

            # Create inputs and labels.
            x, y = [], []
            # Splits each joke into seq_len sized sequences of (input, label) pairs
            for joke in data.data['joke']:
                words = joke.split()
                words.append('<period>')
                encoded_words = [self.vocab.word_to_idx[word] for word in words]
                
                for pos in range(0, len(words) - seq_len, 1):
                    sequence = set_padding(encoded_words[pos: pos + seq_len], seq_len, 0)
                    next_word = encoded_words[pos + seq_len]

                    x.append(sequence)
                    y.append(next_word)

                    # Don't make too many sequences with less than seq_len words
                    if len(sequence) < seq_len:
                        break


            # Use words_to_idx since we might manually add extra vocab for semantic purposes.
            self.vocab_size = len(self.vocab.word_to_idx)
            print(len(x))

            x_train_tensor = torch.LongTensor(x).to(self.device)
            y_train_tensor = torch.LongTensor(y).to(self.device)

            # Convert the processed data into DataLoader class so that Method_RNN can use it easily.
            train_dataset = Model_Dataset(x_train_tensor, y_train_tensor)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader= None

            return {'train': train_loader, 'test': test_loader, 'vocab': self.vocab}
