from src.base_class.method import method
from src.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import pickle
import matplotlib.pyplot as plt

class Method_CNN(method, nn.Module):
    data = None

    # MNIST: n = 2, l_r = 1e-3
    # ORL: n = 10, l_r = 1e-3
    # CIFAR: n = 10, l_r = 1e-3
    max_epoch = 10
    learning_rate = 1e-3

    def __init__(self, mName, mDescription, DATASET):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        if DATASET == 0:
            # MNIST config (0.99)
            self.max_epoch = 2
            self.conv1 = nn.Conv2d(1, 6, 5, dtype=torch.double)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5, dtype=torch.double)
            self.fc1 = nn.Linear(16*4*4, 120, dtype=torch.double)
            self.fc2 = nn.Linear(120, 84, dtype=torch.double)
            self.fc3 = nn.Linear(84, 10, dtype=torch.double)
        elif DATASET == 1:
            # ORL (0.925 w/ batches of 45)
            self.max_epoch = 10
            self.conv1 = nn.Conv2d(3, 6, 5, dtype=torch.double)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5, dtype=torch.double)
            self.fc1 = nn.Linear(16*500, 120, dtype=torch.double)
            self.fc2 = nn.Linear(120, 84, dtype=torch.double)
            self.fc3 = nn.Linear(84, 41, dtype=torch.double)
        elif DATASET == 2:
            # CIFAR (0.73)
            self.max_epoch = 10
            self.conv1 = nn.Conv2d(3, 64, 5, dtype=torch.double)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(64, 128, 5, dtype=torch.double)
            self.fc1 = nn.Linear(128*5*5, 120, dtype=torch.double)
            self.fc2 = nn.Linear(120, 84, dtype=torch.double)
            self.fc3 = nn.Linear(84, 10, dtype=torch.double)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(self.conv1(x))
        # x = self.pool(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # def train(self, X, y):
    def train(self, traindata):
        # optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        for epoch in range(self.max_epoch):
            y_pred, y_true, train_loss = 0, 0, 0
            for i, dataset in enumerate(traindata, 0):
                # tensor = tensor.unsqueeze(1)  # unsqueeze(0) for rgb
                inputs, labels = dataset
                # tensor_x = np.expand_dims(np.array(inputs), 1)

                # y_pred = self(torch.FloatTensor(inputs))
                y_pred = self(inputs.double())
                # y_true = torch.tensor([y[i]])
                y_true = labels

                optimizer.zero_grad()

                train_loss = loss_function(y_pred, y_true)
                train_loss.backward()
                optimizer.step()

                # if i%10 == 0:
                #     accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                #     print('Epoch:', epoch, 'i:',i, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

            # if epoch % 10 == 0:    # print every z mini-batches
            accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
            accuracy, mean_score, std_score, avg_precision, std_precision, avg_recall, std_recall, avg_f1, std_f1 = accuracy_evaluator.evaluate()
            print('Epoch:', epoch, 'Loss:', train_loss.item())
            print('CNN Accuracy: ' + str(accuracy))
            print('CNN Precision: ' + str(avg_precision) + ' +/- ' + str(std_precision))
            print('CNN Recall: ' + str(avg_recall) + ' +/- ' + str(std_recall))
            print('CNN F1: ' + str(avg_f1) + ' +/- ' + str(std_f1))
        print('Finished Training')
    
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

        # dataiter = iter(testdata)
        # images, _ = dataiter.next()
        # outputs = self(images)
        # _, predicted = torch.max(outputs, 1)
        #
        # return predicted

        # y_pred = self(torch.FloatTensor(np.array(X)))
        # return y_pred.max(1)[1]
        

    def run(self):
        print('method running...')
        print('--start training...')
        # self.train(self.data['train']['X'], self.data['train']['y'])
        self.train(self.data['train'])
        print('--start testing...')
        # pred_y = self.test(self.data['test']['X'])
        pred_y, y_true = self.test(self.data['test'])
        # return {'pred_y': pred_y.cpu(), 'true_y': self.data['test']['y']}
        # return {'pred_y': pred_y, 'true_y': self.data['test']}
        return {'pred_y': pred_y, 'true_y': y_true}
