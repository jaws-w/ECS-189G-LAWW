'''
Preprocesses the data for stage 4
'''

import imp
import os
import string
import csv
import re
# import random
# from multiprocessing import Process
import threading

# @returns [ {score: int, words: [words]}, ... ]
def clean_classification_text(dataset_source_folder_path, reviews_dir_path, pos, out, MAX_WORDS):
    print("Reading", reviews_dir_path)
    table = str.maketrans('', '', string.punctuation)
    # cleaned_text_objs = []

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
        out.append((1 if pos else 0, stripped))

    print('Finished', reviews_dir_path)

    # return cleaned_text_objs

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


if __name__ == '__main__':
    # CLASSIFICATION: 0, GENERATION: 1
    DATASET = 0
    MAX_WORDS = 200

    vocab = set()

    if DATASET == 0:
        dataset_source_folder_path = '../../data/stage_4_data/text_classification/'
        # data_obj.dataset_source_file_name = ''

        print('loading data...')
        
        # read the reviews from files and clean them
        # train_neg_txt = clean_classification_text('train/neg/', False)
        # train_pos_txt = clean_classification_text('train/pos/', True)
        # test_neg_txt = clean_classification_text('test/neg/', False)
        # test_pos_txt = clean_classification_text('test/pos/', True)

        train_neg_txt, train_pos_txt, test_neg_txt, test_pos_txt = [], [], [], []

        procs = [threading.Thread(target=clean_classification_text, args=(dataset_source_folder_path, 'train/neg/', False, train_neg_txt, MAX_WORDS)),
            threading.Thread(target=clean_classification_text, args=(dataset_source_folder_path, 'train/pos/', False, train_pos_txt, MAX_WORDS)),
            threading.Thread(target=clean_classification_text, args=(dataset_source_folder_path, 'test/neg/', False, test_neg_txt, MAX_WORDS)),
            threading.Thread(target=clean_classification_text, args=(dataset_source_folder_path, 'test/pos/', False, test_pos_txt, MAX_WORDS))]

        for proc in procs:
            proc.start()

        for proc in procs:
            proc.join()

        # p_train_neg = Process(target=clean_classification_text, args=('train/neg/', False, train_neg_txt,))
        # p_train_pos = Process(target=clean_classification_text, args=('train/pos/', False, train_pos_txt,))
        # p_test_neg = Process(target=clean_classification_text, args=('test/neg/', False, test_neg_txt,))
        # p_test_pos = Process(target=clean_classification_text, args=('test/pos/', False, test_pos_txt,))



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