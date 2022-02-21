'''
Concrete IO class for a specific dataset
'''
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from src.base_class.dataset import dataset
from src.base_class.preprocess_helpers import ConstructVocab, set_tensor_padding
import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab, build_vocab_from_iterator
import pandas as pd

MAX_WORDS = 200

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

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    max_length = 0
    vocab = set()

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):

        print('loading data...')
        
        if self.dataset_name == 'CLASSIFICATION':
            train_data = Classification_Dataset(self.dataset_source_folder_path + '/train.csv')
            test_data = Classification_Dataset(self.dataset_source_folder_path + '/test.csv')

            inputs = ConstructVocab(train_data.data['review'].values.tolist())
            input_tensor = [[inputs.word_to_idx[word] for word in review] for review in train_data.data['review']]

            # add padding
            max_len_input = max(len(i) for i in input_tensor)
            input_tensor = set_tensor_padding(input_tensor, max_len_input)
            print(input_tensor)


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
            pass
        elif self.dataset_name == 'GENERATION':
            pass

