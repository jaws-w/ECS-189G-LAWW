'''
Preprocesses the data for stage 4
'''

import os
import string
import csv
# import random
        
# @returns [ {score: int, words: [words]}, ... ]
def clean_classification_text(reviews_dir_path):
    print("Reading", reviews_dir_path)
    table = str.maketrans('', '', string.punctuation)
    cleaned_text_objs = []

    for file in os.listdir(dataset_source_folder_path + reviews_dir_path):
        # print(file)
        score = int(file[-6:-4].replace('_',''))

        with open(dataset_source_folder_path + reviews_dir_path + file, 'r', encoding='UTF-8') as f:
            try:
                txt = f.read()
            except UnicodeDecodeError:
                print(txt)
                pass

        words = txt.split(maxsplit=MAX_WORDS)
        stripped = [w.translate(table).lower() for w in words[:MAX_WORDS]]
        # stripped = ' '.join(words[:MAX_WORDS])
        # # Get length of longest review for encoding.
        # if len(stripped) > self.max_length:
        #     self.max_length = len(stripped)
        # print(stripped)
        cleaned_text_objs.append((score, stripped))

    return cleaned_text_objs

if 1:
    # CLASSIFICATION: 0, GENERATION: 1
    DATASET = 0
    MAX_WORDS = 200

    vocab = set()

    if DATASET == 0:
        dataset_source_folder_path = '../../data/stage_4_data/text_classification'
        # data_obj.dataset_source_file_name = ''

        print('loading data...')
        
        # read the reviews from files and clean them
        train_neg_txt = clean_classification_text('/train/neg/')
        train_pos_txt = clean_classification_text('/train/pos/')
        test_neg_txt = clean_classification_text('/test/neg/')
        test_pos_txt = clean_classification_text('/test/pos/')

        train_txt = train_neg_txt + train_pos_txt
        test_txt = test_neg_txt + test_pos_txt

        fields = ['rating', 'review']

        # write to train.csv and test.csv
        with open(dataset_source_folder_path + '/train.csv', 'w', newline='', encoding='UTF-8') as train_csv:
            train_writer = csv.writer(train_csv)
            train_writer.writerow(fields)
            train_writer.writerows(train_txt)

        with open(dataset_source_folder_path + '/test.csv', 'w', newline='', encoding='UTF-8') as test_csv:
            train_writer = csv.writer(test_csv)
            train_writer.writerow(fields)
            train_writer.writerows(test_txt)




    elif DATASET == 1:
        pass
        # data_obj = Dataset_Loader('GENERATION', '')
        # data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation'
        # data_obj.dataset_source_file_name = 'ORL'

