from src.base_class.method import method
from src.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import pickle
import matplotlib.pyplot as plt


# We used the readings and tutorial at: https://blog.floydhub.com/gru-with-pytorch/ to learn more about GRU implementation.
class Method_RNN(method, nn.Module):

    data = None
    batch_size = None
    vocab_input_size = None
    out_size = 10
    input_dim = 200

    embedding = None
    fc = None

    # CLASSIFICATION: n = 2, l_r = 1e-3
    # GENERATION: n = 10, l_r = 1e-3
    max_epoch = 10
    learning_rate = 1e-2

    def __init__(self, mName, mDescription, DATASET):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        if DATASET == 0:
            # CLASSIFICATION config
            self.max_epoch = 500

            self.hidden_dim = 16
            self.n_layers = 2
            dropout = 0.2

            # self.dropout = nn.Dropout(p=0.5)
            self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True, dropout=dropout)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(self.hidden_dim, self.out_size)
        elif DATASET == 1:
            # GENERATION config
            # TODO: implement configuration for generation dataset
            pass

    def forward(self, x, h):
        # x = self.embedding(x)
        # self.hidden = self.init_hidden()
        out, h = self.rnn(x, h)
        # out = out[-1, :, :]
        # out = self.dropout(out)
        out = self.fc1(self.relu(out[:,-1]))

        return out, h

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        return hidden

    # def train(self, X, y):
    def train(self, traindata):
        # optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        for epoch in range(self.max_epoch):
            y_pred, y_true, train_loss = 0, 0, 0

            h = self.init_hidden()

            for i, dataset in enumerate(traindata, 0):
                inputs, labels = dataset
                
                h = h.data

                y_pred, h = self(inputs.view(self.batch_size, -1, self.input_dim), h)

                y_true = labels

                optimizer.zero_grad()

                train_loss = loss_function(y_pred, y_true)
                train_loss.backward()
                optimizer.step()

            # if epoch % 10 == 0:    # print every z mini-batches
            accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
            accuracy, mean_score, std_score, avg_precision, std_precision, avg_recall, std_recall, avg_f1, std_f1 = accuracy_evaluator.evaluate()
            print('Epoch:', epoch, 'Loss:', train_loss.item())
            print('RNN Accuracy: ' + str(accuracy))
            print('RNN Precision: ' + str(avg_precision) + ' +/- ' + str(std_precision))
            print('RNN Recall: ' + str(avg_recall) + ' +/- ' + str(std_recall))
            print('RNN F1: ' + str(avg_f1) + ' +/- ' + str(std_f1))
        print('Finished Training')
    
    def test(self, testdata):
        for _, dataset in enumerate(testdata, 0):
            inputs, labels = dataset
            outputs, _ = self(inputs.view(self.batch_size, -1, self.input_dim), self.init_hidden())

            # for i in range(5):
            #     # plt.imshow(testdata.data['test'][i]['image'], cmap="Greys")
            #     # plt.show()
            #     print('pred: ', outputs.max(1)[1][i])
            #     print('true: ', y_true[i])
            return outputs.max(1)[1], labels
        

    def run(self):
        print('method running...')
        print('--start training...')
        self.embedding = nn.Embedding(self.vocab_input_size, self.input_dim)
        self.fc = nn.Linear(self.hidden_dim, self.out_size, dtype=torch.double)

        self.train(self.data['train'])
        print('--start testing...')
        pred_y, y_true = self.test(self.data['test'])
        return {'pred_y': pred_y, 'true_y': y_true}
