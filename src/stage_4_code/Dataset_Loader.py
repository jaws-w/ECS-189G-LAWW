'''
Concrete IO class for a specific dataset
'''
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from copyreg import clear_extension_cache
from src.base_class.dataset import dataset

import os
import string
import random

MAX_WORDS  = 200

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

            train_neg_txt = self.clean_classification_text('/train/neg/')
            train_pos_txt = self.clean_classification_text('/train/pos/')
            test_neg_txt = self.clean_classification_text('/test/neg/')
            test_pos_txt = self.clean_classification_text('/test/pos/')
            
            print(len(self.vocab))

            train_txt = train_neg_txt + train_pos_txt
            test_txt = test_neg_txt + test_pos_txt

            random.shuffle(train_txt)
            random.shuffle(test_txt)
            return {'train': train_txt, 'test': test_txt}

        elif self.dataset_name == 'GENERATION':
            pass

    # @returns [ {words: [words], score: int}, ... ]
    def clean_classification_text(self, reviews_dir_path):
        print("Reading", reviews_dir_path)
        table = str.maketrans('', '', string.punctuation)
        cleaned_text_objs = []

        for file in os.listdir(self.dataset_source_folder_path + reviews_dir_path):
            # print(file)
            
            score = int(file[-5:-4])

            f = open(self.dataset_source_folder_path + reviews_dir_path + file, 'r', encoding='UTF-8')
            try:
                txt = f.read()
            except UnicodeDecodeError:
                # print(file)
                pass

            f.close()

            words = txt.split()
            stripped = [w.translate(table).lower() for w in words[:MAX_WORDS]]

            # Get length of longest review for encoding.
            if len(stripped) > self.max_length:
                self.max_length = len(stripped)
            # print(stripped)
            cleaned_text_objs.append((stripped, score))
            self.vocab = self.vocab.union(set(stripped))

        return cleaned_text_objs
