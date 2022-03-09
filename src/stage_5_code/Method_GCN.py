from base_class.method import method
from stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
# from src.base_class.method import method
# from src.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# We used the readings and tutorial at: https://blog.floydhub.com/gru-with-pytorch/ to learn more about GRU implementation.
class Method_GCN(method, nn.Module):

    def __init__(self, mName, mDescription, DATASET):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.DATASET = DATASET
        self.out_size = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if DATASET == 0:
            # CORA config
            # self.max_epoch = 50  # 67-70%
            # self.learning_rate = 1e-2
            #
            # self.conv1 = GCNConv(1433, 16)
            # self.conv2 = GCNConv(16, 7)  # cora has 7 different classes
            # self.fc = nn.Linear(7, 7)

            # self.max_epoch = 25  # 73-75%
            # self.learning_rate = 1e-3
            #
            # self.conv1 = GCNConv(1433, 512)
            # self.conv2 = GCNConv(512, 256)  # cora has 7 different classes
            # self.fc = nn.Linear(256, 7)

            self.max_epoch = 25  # >>76%<<
            self.learning_rate = 1e-3

            self.conv1 = GCNConv(1433, 1024)
            self.conv2 = GCNConv(1024, 512)  # cora has 7 different classes
            self.fc = nn.Linear(512, 7)
        elif DATASET == 1:
            # CITESEER config
            # self.max_epoch = 50  # 57-61%
            # self.learning_rate = 5e-3
            #
            # self.conv1 = GCNConv(3703, 16)
            # self.conv2 = GCNConv(16, 6)  # citeseer has 6 different classes
            # self.fc = nn.Linear(6, 6)

            # self.max_epoch = 20  # 64% at 50 epochs 5e-3, 65-66% at 20 epochs 1e-3, 65% at 30epochs, 1e-3
            # self.learning_rate = 1e-3
            #
            # self.conv1 = GCNConv(3703, 1024)
            # self.conv2 = GCNConv(1024, 512)  # citeseer has 6 different classes
            # self.fc = nn.Linear(512, 6)

            self.max_epoch = 40  # 63% at 20 epochs, 65% at 25 epochs, >>66% at 30 epochs<<, 64% at 40 epochs
            self.learning_rate = 1e-3

            self.conv1 = GCNConv(3703, 512)
            self.conv2 = GCNConv(512, 64)  # citeseer has 6 different classes
            self.fc = nn.Linear(64, 6)
        elif DATASET == 2:
            # PUBMED config
            # self.max_epoch = 250  # 67% at 400 epoch, 66% at 250
            # self.learning_rate = 1e-3
            #
            # self.conv1 = GCNConv(500, 16)
            # self.conv2 = GCNConv(16, 3)  # citeseer has 6 different classes
            # self.fc = nn.Linear(3, 3)

            self.max_epoch = 150  # 66% at 100 epoch, 68-71% at 125 epoch, >>70-72% at 150 epoch<<
            self.learning_rate = 1e-3

            self.conv1 = GCNConv(500, 128)
            self.conv2 = GCNConv(128, 32)  # citeseer has 6 different classes
            self.fc = nn.Linear(32, 3)
        elif DATASET == 3:
            # DEBUG config (mini-CORA)
            self.max_epoch = 500
            self.learning_rate = 1e-3

            self.conv1 = GCNConv(1433, 16)
            self.conv2 = GCNConv(16, 7)

    def forward(self, traindata):
        x, edge_index = traindata.x, traindata.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # return self.fc(F.relu(x))
        return F.log_softmax(x, dim=1)

    def do_train_cora(self, traindata):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        for epoch in range(self.max_epoch):
            y_pred = self(traindata)
            y_true = traindata.y

            optimizer.zero_grad()
            train_loss = loss_function(y_pred[self.data['train_test_val']['idx_train']], y_true)
            train_loss.backward()
            optimizer.step()

            accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred[self.data['train_test_val']['idx_train']].max(1)[1]}
            accuracy, mean_score, std_score, avg_precision, std_precision, avg_recall, std_recall, avg_f1, std_f1 = accuracy_evaluator.evaluate()
            print('Epoch:', epoch + 1, 'Loss:', train_loss.item())
            print('RNN Accuracy: ' + str(accuracy))
            print('RNN Precision: ' + str(avg_precision) + ' +/- ' + str(std_precision))
            print('RNN Recall: ' + str(avg_recall) + ' +/- ' + str(std_recall))
            print('RNN F1: ' + str(avg_f1) + ' +/- ' + str(std_f1))
        print('Finished Training')

    def train_data(self, traindata):
        self.do_train_cora(traindata)
    
    def test(self, testdata):
        pred_y = self(testdata)
        return pred_y[self.data['train_test_val']['idx_test']].max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        # self.embedding = nn.Embedding(self.vocab_input_size, self.input_dim).to(self.device)

        edge_idx = torch.LongTensor(self.data['graph']['edge']).t().contiguous()
        traindata = Data(
            x=self.data['graph']['X'],
            edge_index=edge_idx,
            # edge_attr=self.data['graph']['X'],
            y=self.data['graph']['y'][self.data['train_test_val']['idx_train']],
            pos=self.data['train_test_val']['idx_train']
        )
        testdata = Data(
            x=self.data['graph']['X'],
            edge_index=edge_idx,
            # edge_attr=self.data['graph']['X'],
            y=self.data['graph']['y'][self.data['train_test_val']['idx_test']],
            pos=self.data['train_test_val']['idx_train']
        )

        self.train_data(traindata)
        print('--start testing...')
        pred_y = self.test(testdata)
        y_true = self.data['graph']['y'][self.data['train_test_val']['idx_test']]
        return {'pred_y': pred_y, 'true_y': y_true}
