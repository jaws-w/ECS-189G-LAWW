'''
Concrete IO class for a specific dataset
'''
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from src.base_class.dataset import dataset
from src.base_class.preprocess_helpers import ConstructVocab, set_tensor_padding

import torch
from torch.utils.data import Dataset, DataLoader
# from torchtext.vocab import Vocab, build_vocab_from_iterator

import numpy as np
import pandas as pd

MAX_WORDS = 200


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
        self.length = [np.sum(1 - np.equal(x, 0)) for x in inputs]

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]

        return x, y, x_len

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
        self.batch_size = batch_size
    
    def load(self):

        print('loading data...')
        
        if self.dataset_name == 'CLASSIFICATION':
            train_data = Classification_Dataset(self.dataset_source_folder_path + '/train.csv')
            test_data = Classification_Dataset(self.dataset_source_folder_path + '/test.csv')

            all_data = train_data.data['review'] + test_data.data['review']
            inputs = ConstructVocab(all_data.values.tolist())
            self.vocab_size = len(inputs.word_to_idx)

            # inputs_tensor = [[inputs.word_to_idx[word] for word in review] for review in all_data]
            #
            # max_inputs_length = max(len(i) for i in inputs_tensor)
            # full_dataset_tensor = set_tensor_padding(inputs_tensor, max_inputs_length)

            self.label_size = train_data.data['rating'][0]

            # Create vocab from the text in the reviews.
            train_inputs = ConstructVocab(train_data.data['review'].values.tolist())
            train_tensor = [[train_inputs.word_to_idx[word] for word in review] for review in train_data.data['review']]

            test_inputs = ConstructVocab(test_data.data['review'].values.tolist())
            test_tensor = [[test_inputs.word_to_idx[word] for word in review] for review in test_data.data['review']]

            # Add padding.
            max_len_train_input = max(len(i) for i in train_tensor)
            train_tensor = set_tensor_padding(train_tensor, max_len_train_input)
            # print(train_tensor)

            max_len_test_input = max(len(i) for i in test_tensor)
            test_tensor = set_tensor_padding(test_tensor, max_len_test_input)

            # Convert the processed data into DataLoader class so that Method_RNN can use it easily.
            train_dataset = Model_Dataset(train_tensor, train_data.data['rating'])
            test_dataset = Model_Dataset(test_tensor, test_data.data['rating'])

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

            return {'train': train_loader, 'test': test_loader}

            # train_neg_txt = self.clean_classification_text('/train/neg/')
            # train_pos_txt = self.clean_classification_text('/train/pos/')
            # test_neg_txt = self.clean_classification_text('/test/neg/')
            # test_pos_txt = self.clean_classification_text('/test/pos/')
            
            # print(len(self.vocab))

            # train_txt = train_neg_txt + train_pos_txt
            # test_txt = test_neg_txt + test_pos_txt

            # random.shuffle(train_txt)
            # random.shuffle(test_txt)
            # return {'train': train_txt, 'test': test_txt}
            # pass
        elif self.dataset_name == 'GENERATION':
            pass

