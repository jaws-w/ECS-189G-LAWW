'''
Preprocesses the data for stage 4
'''

import os
import string
import csv
import re
# import random

# @returns [ {score: int, words: [words]}, ... ]
def clean_classification_text(reviews_dir_path, pos):
    print("Reading", reviews_dir_path)
    table = str.maketrans('', '', string.punctuation)
    cleaned_text_objs = []

    for file in os.listdir(dataset_source_folder_path + reviews_dir_path):
        # print(file)
        # score = int(file[-6:-4].replace('_',''))

        with open(dataset_source_folder_path + reviews_dir_path + file, 'r', encoding='UTF-8') as f:
            try:
                txt = f.read()
            except UnicodeDecodeError:
                print(txt)
                pass

        words = txt.split(maxsplit=MAX_WORDS)
        stripped = [w.translate(table).lower() for w in words[:MAX_WORDS]]
        stripped = ' '.join(stripped[:MAX_WORDS])
        # # Get length of longest review for encoding.
        # if len(stripped) > self.max_length:
        #     self.max_length = len(stripped)
        # print(stripped)
        cleaned_text_objs.append((1 if pos else 0, stripped))

    return cleaned_text_objs

def clean_generation_text(file_path):
    print("Reading", file_path)

    table = str.maketrans('', '', string.punctuation)

    cleaned_jokes = None

    with open(file_path, 'r', encoding='UTF-8') as f:
        jokes = f.readlines()[1:]
        jokes = [joke[re.search(r',', joke).start() + 2:-2] for joke in jokes]
        jokes = [' '.join([w.translate(table).lower() for w in joke.split()]) for joke in jokes]

    cleaned_jokes = [(i, jokes[i]) for i in range(len(jokes))]

    return cleaned_jokes


if 1:
    # CLASSIFICATION: 0, GENERATION: 1
    DATASET = 0
    MAX_WORDS = 500

    vocab = set()

    if DATASET == 0:
        dataset_source_folder_path = '../../data/stage_4_data/text_classification/'
        # data_obj.dataset_source_file_name = ''

        print('loading data...')

        # read the reviews from files and clean them
        train_neg_txt = clean_classification_text('train/neg/', False)
        train_pos_txt = clean_classification_text('train/pos/', True)
        test_neg_txt = clean_classification_text('test/neg/', False)
        test_pos_txt = clean_classification_text('test/pos/', True)

        train_txt = train_neg_txt + train_pos_txt
        test_txt = test_neg_txt + test_pos_txt

        fields = ['rating', 'review']

        # write to train.csv and test.csv
        print("Writing")
        with open(dataset_source_folder_path + '/train.csv', 'w', newline='', encoding='UTF-8') as train_csv:
            train_writer = csv.writer(train_csv)
            train_writer.writerow(fields)
            train_writer.writerows(train_txt)

        with open(dataset_source_folder_path + '/test.csv', 'w', newline='', encoding='UTF-8') as test_csv:
            train_writer = csv.writer(test_csv)
            train_writer.writerow(fields)
            train_writer.writerows(test_txt)




    elif DATASET == 1:
        dataset_source_folder_path = '../../data/stage_4_data/text_generation/'
        dataset_source_file_name = 'data'

        print('loading data...')

        train_txt = clean_generation_text(dataset_source_folder_path + dataset_source_file_name)

        print('writing...')

        with open(dataset_source_folder_path + 'cleaned.csv', 'w', newline='', encoding='UTF-8') as f:
            train_writer = csv.writer(f)
            train_writer.writerow(['id', 'joke'])
            train_writer.writerows(train_txt)