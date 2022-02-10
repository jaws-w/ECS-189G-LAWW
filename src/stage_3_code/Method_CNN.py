from src.base_class.method import method
from src.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Method_CNN(method, nn.Module):
    data = None

    max_epoch = 10
    learning_rate = 1e-3

    batch_size = 4

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(1, 6, 5, dtype=torch.double)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, dtype=torch.double)
        self.fc1 = nn.Linear(16*4*4, 120, dtype=torch.double)
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
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        for epoch in range(self.max_epoch):
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

                # if epoch%100 == 0:
                #     accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                #     print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

                if i % 2000 == 1999:    # print every 2000 mini-batches
                    accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                    print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
        print('Finished Training')
    
    def test(self, testdata):
        dataiter = iter(testdata)
        images, _ = dataiter.next()
        outputs = self(images)
        _, predicted = torch.max(outputs, 1)

        return predicted
        # y_pred = self(torch.FloatTensor(np.array(X)))
        # return y_pred.max(1)[1]
        

    def run(self):
        print('method running...')
        print('--start training...')
        # self.train(self.data['train']['X'], self.data['train']['y'])
        self.train(self.data['train'])
        print('--start testing...')
        # pred_y = self.test(self.data['test']['X'])
        pred_y = self.test(self.data['test'])
        # return {'pred_y': pred_y.cpu(), 'true_y': self.data['test']['y']}
        return {'pred_y': pred_y.cpu(), 'true_y': self.data['test']}