'''
Concrete IO class for a specific dataset
'''
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from cProfile import label
from base_class.dataset import dataset
from base_class.preprocess_helpers import ConstructVocab, set_tensor_padding, set_padding

import math

import torch
from torch.utils.data import Dataset, DataLoader
# from torchtext.vocab import Vocab, build_vocab_from_iterator

import numpy as np
import pandas as pd
import random

# MAX_WORDS = 200


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
        # self.length = [np.sum(1 - np.equal(x, 0)) for x in inputs]

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        # x_len = self.length[index]

        return x, y#, x_len

    def __len__(self):
        return len(self.data)


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    label_size = None
    vocab_size = None
    vocab = set()

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
            inputs = ConstructVocab(all_data)
            self.vocab_size = len(inputs.word_to_idx)

            self.label_size = train_data.data['rating'][0]

            # Create input tensors.
            train_tensor = [[inputs.word_to_idx[word] for word in review.split()] for review in train_data.data['review']]
            test_tensor = [[inputs.word_to_idx[word] for word in review.split()] for review in test_data.data['review']]

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
            seq_len = 5
            self.label_size = seq_len

            data = Classification_Dataset(self.dataset_source_folder_path + '/cleaned.csv')
            jokes_obj = ConstructVocab(data.data['joke'])

            # Create inputs and labels.
            x, y = [], []
            # Splits each joke into seq_len sized sequences of (input, label) pairs
            for joke in data.data['joke']:
                words = joke.split()
                words.append('<period>')
                encoded_words = [jokes_obj.word_to_idx[word] for word in words]
                for i in range(math.ceil(len(encoded_words)/seq_len)):
                    next_pos = i*seq_len
                    sequence = encoded_words[next_pos: next_pos+seq_len]
                    if len(sequence) < seq_len:
                        sequence = set_padding(sequence, seq_len, 0)
                    joke_input = sequence[:-1]
                    joke_output = sequence[1:]
                    x.append(joke_input)
                    y.append(joke_output)

                    # Don't make too many sequences with less than seq_len words
                    if len(sequence) < seq_len:
                        break

            # Use words_to_idx since we might manually add extra vocab for semantic purposes.
            self.vocab_size = len(jokes_obj.word_to_idx)

            x_train_tensor = torch.LongTensor(x).to(self.device)
            y_train_tensor = torch.LongTensor(y).to(self.device)

            # Does not need a target for testing set.
            rand_sample = randomSample(0, len(data.data['joke']), 10)
            sample_jokes = [data.data['joke'][i].split()[:3] for i in rand_sample]
            x_test_sample = []
            for joke in sample_jokes:
                x_test_sample.append([jokes_obj.word_to_idx[word] for word in joke])
            # x_test_sample = [jokes_obj.word_to_idx[word] for word in sample_words]
            x_test_tensor = torch.LongTensor(x_test_sample).to(self.device)

            # Convert the processed data into DataLoader class so that Method_RNN can use it easily.
            train_dataset = Model_Dataset(x_train_tensor, y_train_tensor)
            test_dataset = Model_Dataset(x_test_tensor, x_test_tensor)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            return {'train': train_loader, 'test': test_loader}

            # first_n_words = 3
            # data = Classification_Dataset(self.dataset_source_folder_path + '/cleaned.csv')
            #
            # jokes_obj = ConstructVocab(data.data['joke'])
            #
            # # Create inputs and labels.
            # x, y = [], []
            # for joke in data.data['joke']:
            #     words = joke.split()
            #     # x[len(x):] = [jokes_obj.word_to_idx[word] for word in words[:first_n_words]]
            #     # y[len(y):] = [jokes_obj.word_to_idx[word] for word in words[first_n_words:]]
            #     x.append([jokes_obj.word_to_idx[word] for word in words[:first_n_words]])
            #     y.append([jokes_obj.word_to_idx[word] for word in words[first_n_words:]])
            #
            # self.vocab_size = len(jokes_obj.word_to_idx)
            #
            # # Establish padding size for labels.
            # self.label_size = max(len(joke_ending) for joke_ending in y)
            #
            # # Create training and testing tensors and add padding.
            # max_len_input = first_n_words
            # x_train_tensor = torch.LongTensor(set_tensor_padding(x, max_len_input)).to(self.device)
            # y_train_tensor = torch.LongTensor(set_tensor_padding(y, self.label_size)).to(self.device)
            #
            # rand_sample = randomSample(0, len(x), 10)
            # x_test = [x[i] for i in rand_sample]
            # y_test = [y[i] for i in rand_sample]
            # x_test_tensor = torch.LongTensor(set_tensor_padding(x_test, max_len_input)).to(self.device)
            # y_test_tensor = torch.LongTensor(set_tensor_padding(y_test, self.label_size)).to(self.device)
            #
            # # Convert the processed data into DataLoader class so that Method_RNN can use it easily.
            # train_dataset = Model_Dataset(x_train_tensor, y_train_tensor)
            # test_dataset = Model_Dataset(x_test_tensor, y_test_tensor)
            #
            # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            # test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            #
            # return {'train': train_loader, 'test': test_loader}

def randomSample(start, end, amount):
    return [random.randint(start, end) for _ in range(amount)]
