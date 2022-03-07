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
            self.max_epoch = 20  # 85%
            self.learning_rate = 1e-3

            self.n_layers = 2
            self.dropout = 0.2
            self.out_size = 2

            self.input_dim = 50
            self.hidden_dim = 50
            self.input_dim1 = 50
            self.hidden_dim1 = 50

            self.conv1 = None
            self.conv2 = None
            self.relu = nn.ReLU().to(self.device)
            self.fc = nn.Linear(self.hidden_dim1, self.out_size).to(self.device)
        elif DATASET == 1:
            # CITESEER config
            pass
        elif DATASET == 2:
            # PUBMED config
            pass
        elif DATASET == 3:
            # DEBUG config (mini-CORA)
            self.max_epoch = 20  # 85%
            self.learning_rate = 1e-3

            self.conv1 = None
            self.conv2 = None

    def forward(self, traindata):
        x, edge_index = traindata.x, traindata.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

    def do_train_cora(self, traindata):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        for epoch in range(self.max_epoch):
            y_pred = self(traindata)
            y_true = traindata.y

            optimizer.zero_grad()
            train_loss = loss_function(y_pred, y_true)
            train_loss.backward()
            optimizer.step()

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
            pass
        elif self.DATASET == 1:
            pass
        elif self.DATASET == 2:
            pass
        elif self.DATASET == 3:
            self.do_train_cora(traindata)
    
    def test(self, testdata):
        if self.DATASET == 0:
            pass
        elif self.DATASET == 1:
            pass
        elif self.DATASET == 2:
            pass
        elif self.DATASET == 3:
            pass

    def run(self):
        print('method running...')
        print('--start training...')
        # self.embedding = nn.Embedding(self.vocab_input_size, self.input_dim).to(self.device)
        # if self.DATASET == 1:
        #     self.fc = nn.Linear(self.hidden_dim1, self.vocab_input_size).to(self.device)

        edge_idx = torch.LongTensor(self.data['graph']['edge']).t().contiguous()
        traindata = Data(
            x=self.data['graph']['X'],
            edge_index=edge_idx,
            # edge_attr=self.data['graph']['X'],
            y=self.data['graph']['y'][self.data['train_test_val']['idx_train']],
            pos=self.data['train_test_val']['idx_train']
        )

        if self.DATASET == 0:
            pass
        elif self.DATASET == 1:
            pass
        elif self.DATASET == 2:
            pass
        elif self.DATASET == 3:
            # Debug cora
            self.conv1 = GCNConv(1433, 16)
            self.conv2 = GCNConv(16, 7)  # cora has 7 different classes

        self.train_data(traindata)
        print('--start testing...')
        if self.DATASET == 0:
            pred_y, y_true = self.test(self.data['test'])
            return {'pred_y': pred_y, 'true_y': y_true}
