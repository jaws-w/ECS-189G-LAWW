# Text-preprocessing code adapted from text-preprocessing tutorial at:
# https://medium.com/swlh/tensorflow-vs-pytorch-for-text-classification-using-gru-e95f1b68fa2d

import numpy as np


# Construct a vocabulary
class ConstructVocab:
    def __init__(self, text_entries):
        self.text_entries = text_entries
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = set()
        self.create_index()

    def create_index(self):
        # for entry in self.text_entries:
        #     self.vocab.update(entry)
        for entry in self.text_entries:
            # print(entry)
            self.vocab.update(entry.split())
            pass

        # sort vocabulary
        self.vocab = sorted(self.vocab)

        # add a padding token with index 0
        self.word_to_idx['<pad>'] = 0

        # word to index mapping
        for index, word in enumerate(self.vocab):
            self.word_to_idx[word] = index + 1  # +1 offset since 0 is the pad

        # index to word mapping
        for word, index in self.word_to_idx.items():
            self.idx_to_word[index] = word


# Do padding calculation
def do_padding(lst, max_len):
    padded = np.zeros((max_len), dtype=np.int64)

    if len(lst) > max_len:
        padded[:] = lst[:max_len]
    else:
        padded[:len(lst)] = lst

    return padded


# Sets padding to all elements in given tensor.
def set_tensor_padding(tensor, max_len):
    return np.array([do_padding(x, max_len) for x in tensor])
