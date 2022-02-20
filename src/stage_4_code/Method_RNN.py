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

    # CLASSIFICATION: n = 2, l_r = 1e-3
    # GENERATION: n = 10, l_r = 1e-3
    max_epoch = 10
    learning_rate = 1e-2

    def __init__(self, mName, mDescription, DATASET, mData):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.data = mData

        if DATASET == 0:
            # CLASSIFICATION config
            self.max_epoch = 10

            hidden_dim = 16
            n_layers = 2

            self.rnn = nn.GRU(self.data.max_length, hidden_dim, n_layers)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(hidden_dim, output_dim, dtype=torch.double)
        elif DATASET == 1:
            # GENERATION config
            # TODO: implement configuration for generation dataset
            pass

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(self.relu(out[:,-1]))
        return x

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden

    # def train(self, X, y):
    # def train(self, traindata):
    #     # optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
    #     optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    #     loss_function = nn.CrossEntropyLoss()
    #     accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

    #     for epoch in range(self.max_epoch):
    #         y_pred, y_true, train_loss = 0, 0, 0
    #         for i, dataset in enumerate(traindata, 0):
    #             # tensor = tensor.unsqueeze(1)  # unsqueeze(0) for rgb
    #             inputs, labels = dataset
    #             # tensor_x = np.expand_dims(np.array(inputs), 1)

    #             # y_pred = self(torch.FloatTensor(inputs))
    #             y_pred = self(inputs.double())
    #             # y_true = torch.tensor([y[i]])
    #             y_true = labels

    #             optimizer.zero_grad()

    #             train_loss = loss_function(y_pred, y_true)
    #             train_loss.backward()
    #             optimizer.step()

    #         # if epoch % 10 == 0:    # print every z mini-batches
    #         accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
    #         accuracy, mean_score, std_score, avg_precision, std_precision, avg_recall, std_recall, avg_f1, std_f1 = accuracy_evaluator.evaluate()
    #         print('Epoch:', epoch, 'Loss:', train_loss.item())
    #         print('RNN Accuracy: ' + str(accuracy))
    #         print('RNN Precision: ' + str(avg_precision) + ' +/- ' + str(std_precision))
    #         print('RNN Recall: ' + str(avg_recall) + ' +/- ' + str(std_recall))
    #         print('RNN F1: ' + str(avg_f1) + ' +/- ' + str(std_f1))
    #     print('Finished Training')
    
    def test(self, testdata):
        for _, dataset in enumerate(testdata, 0):
            images, y_true = dataset
            outputs = self(images.double())

            # for i in range(5):
            #     # plt.imshow(testdata.data['test'][i]['image'], cmap="Greys")
            #     # plt.show()
            #     print('pred: ', outputs.max(1)[1][i])
            #     print('true: ', y_true[i])
            return outputs.max(1)[1], y_true
        

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train'])
        print('--start testing...')
        pred_y, y_true = self.test(self.data['test'])
        return {'pred_y': pred_y, 'true_y': y_true}
