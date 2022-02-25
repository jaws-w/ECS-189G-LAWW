from base_class.method import method
from stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import pickle
import matplotlib.pyplot as plt


# We used the readings and tutorial at: https://blog.floydhub.com/gru-with-pytorch/ to learn more about GRU implementation.
class Method_RNN(method, nn.Module):

    # data = None
    # batch_size = None
    # vocab_input_size = None
    

    # CLASSIFICATION: n = 2, l_r = 1e-3
    # GENERATION: n = 10, l_r = 1e-3
    

    def __init__(self, mName, mDescription, DATASET):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.DATASET = DATASET

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if DATASET == 0:
            # CLASSIFICATION config
            self.max_epoch = 10  # 85%
            self.learning_rate = 0.5e-2

            self.n_layers = 2
            self.dropout = 0.2
            self.out_size = 2

            self.input_dim = 50
            self.hidden_dim = 50
            self.input_dim1 = 50
            self.hidden_dim1 = 50

            self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True, dropout=self.dropout).to(self.device)
            self.rnn1 = nn.GRU(self.input_dim1, self.hidden_dim1, self.n_layers, batch_first=True, dropout=self.dropout).to(self.device)
            self.relu = nn.ReLU().to(self.device)
            self.fc = nn.Linear(self.hidden_dim1, self.out_size).to(self.device)
        elif DATASET == 1:
            # GENERATION config
            self.max_epoch = 10  # 85%
            self.learning_rate = 0.5e-2

            self.n_layers = 2
            self.dropout = 0.2
            self.out_size = 2

            self.input_dim = 50
            self.hidden_dim = 50
            self.input_dim1 = 50
            self.hidden_dim1 = 50

            self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True,
                              dropout=self.dropout).to(self.device)
            self.rnn1 = nn.GRU(self.input_dim1, self.hidden_dim1, self.n_layers, batch_first=True,
                               dropout=self.dropout).to(self.device)
            self.relu = nn.ReLU().to(self.device)
            self.fc = nn.Linear(self.hidden_dim1, self.out_size).to(self.device)
            pass

    def forward(self, x, h):
        x = self.embedding(x)
        out, h = self.rnn(x, h)
        out, h = self.rnn1(out, h)
        out = self.fc(self.relu(out[:, -1]))

        return out, h

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden

    def doTrainTextClassification(self, traindata):
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

                y_pred, h = self(inputs, h)
                y_true = labels

                optimizer.zero_grad()

                train_loss = loss_function(y_pred, y_true)
                train_loss.backward()
                optimizer.step()

            # if epoch % 10 == 0:    # print every z mini-batches
            accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
            accuracy, mean_score, std_score, avg_precision, std_precision, avg_recall, std_recall, avg_f1, std_f1 = accuracy_evaluator.evaluate()
            print('Epoch:', epoch + 1, 'Loss:', train_loss.item())
            print('RNN Accuracy: ' + str(accuracy))
            print('RNN Precision: ' + str(avg_precision) + ' +/- ' + str(std_precision))
            print('RNN Recall: ' + str(avg_recall) + ' +/- ' + str(std_recall))
            print('RNN F1: ' + str(avg_f1) + ' +/- ' + str(std_f1))
        print('Finished Training')

    def doTrainTextGeneration(self, traindata):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        for epoch in range(self.max_epoch):
            y_pred, y_true, train_loss = 0, 0, 0

            h = self.init_hidden()

            for i, dataset in enumerate(traindata, 0):
                inputs, labels = dataset

                h = h.data

                y_pred, h = self(inputs, h)
                y_true = labels

                optimizer.zero_grad()

                train_loss = loss_function(y_pred, y_true)
                train_loss.backward()
                optimizer.step()

            # if epoch % 10 == 0:    # print every z mini-batches
            accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
            accuracy, mean_score, std_score, avg_precision, std_precision, avg_recall, std_recall, avg_f1, std_f1 = accuracy_evaluator.evaluate()
            print('Epoch:', epoch + 1, 'Loss:', train_loss.item())
            print('RNN Accuracy: ' + str(accuracy))
            print('RNN Precision: ' + str(avg_precision) + ' +/- ' + str(std_precision))
            print('RNN Recall: ' + str(avg_recall) + ' +/- ' + str(std_recall))
            print('RNN F1: ' + str(avg_f1) + ' +/- ' + str(std_f1))
        print('Finished Training')

    def train_data(self, traindata):
        if self.DATASET == 0:
            self.doTrainTextClassification(traindata)
        if self.DATASET == 1:
            self.doTrainTextGeneration(traindata)
    
    def test(self, testdata):
        if self.DATASET == 0:
            self.eval()
            pred_y, true_y = [],[]
            for i, dataset in enumerate(testdata, 0):
                inputs, labels = dataset
                outputs, _ = self(inputs, self.init_hidden())
                pred_y.append(outputs.max(1)[1])
                true_y.append(labels)

            return torch.flatten(torch.stack(pred_y)), torch.flatten(torch.stack(true_y))
        if self.DATASET == 1:
            # TODO: implement test() for text generation.
            Exception('TODO: implement test() for text generation.')
        

    def run(self):
        print('method running...')
        print('--start training...')
        self.embedding = nn.Embedding(self.vocab_input_size, self.input_dim).to(self.device)
        # self.fc = nn.Linear(self.hidden_dim, self.out_size, dtype=torch.double)

        self.train_data(self.data['train'])
        print('--start testing...')
        pred_y, y_true = self.test(self.data['test'])
        return {'pred_y': pred_y, 'true_y': y_true}
