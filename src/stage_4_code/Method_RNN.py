from src.base_class.method import method
from src.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# We used the readings and tutorial at: https://blog.floydhub.com/gru-with-pytorch/ to learn more about GRU implementation.
class Method_RNN(method, nn.Module):

    def __init__(self, mName, mDescription, DATASET):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.DATASET = DATASET
        self.out_size = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if DATASET == 0:
            # CLASSIFICATION config
            self.max_epoch = 20  # 85%
            self.learning_rate = 1e-3

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
            self.max_epoch = 100  # 91%
            self.learning_rate = 5e-3

            self.n_layers = 2
            self.dropout = 0.2
            self.out_size = 1

            x = 512
            self.input_dim = 128
            self.hidden_dim = x
            self.input_dim1 = x
            self.hidden_dim1 = x

            self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True,
                              dropout=self.dropout).to(self.device)
            self.relu = nn.ReLU().to(self.device)
            self.fc = nn.Linear(self.hidden_dim1, self.out_size).to(self.device)
            pass

    def forward(self, x, h):
        x = self.embedding(x)
        out, h = self.rnn(x, h)
        if self.DATASET == 0:
            out, h = self.rnn1(out, h)
        out = self.fc(self.relu(out[:, -1]))

        return out, h

    def init_hidden(self, batch_size=1):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden

    def doTrainTextClassification(self, traindata):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        for epoch in range(self.max_epoch):
            y_pred, y_true, train_loss = 0, 0, 0

            h = self.init_hidden(self.batch_size)

            for i, dataset in enumerate(traindata, 0):
                inputs, labels = dataset

                h = h.data

                y_pred, h = self(inputs, h)
                y_true = labels

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

    def doTrainTextGeneration(self, traindata):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        for epoch in range(self.max_epoch):

            h = self.init_hidden(self.batch_size)

            for i, dataset in enumerate(traindata, 0):
                inputs, labels = dataset

                y_pred, h = self(inputs, h)
                y_true = labels

                optimizer.zero_grad()

                train_loss = loss_function(y_pred, y_true)
                train_loss.backward()
                optimizer.step()

            if epoch % 10 == 0 or epoch == self.max_epoch - 1:    # print every 10 epochs
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                accuracy, mean_score, std_score, avg_precision, std_precision, avg_recall, std_recall, avg_f1, std_f1 = accuracy_evaluator.evaluate()
                print('Epoch:', epoch + 1, 'Loss:', train_loss.item())
                print('RNN Accuracy: ' + str(accuracy))
                print('RNN Precision: ' + str(avg_precision) + ' +/- ' + str(std_precision))
                print('RNN Recall: ' + str(avg_recall) + ' +/- ' + str(std_recall))
                print('RNN F1: ' + str(avg_f1) + ' +/- ' + str(std_f1))
        print('Finished Training')

    def predict_joke(self):
        joke_start = input("Enter a joke beginning : ")
        joke_start = joke_start.split()

        h = self.init_hidden()
        try:
            prompt = [self.data['vocab'].word_to_idx[word] for word in joke_start]
        except KeyError:
            print("Naughty naughty, that's not a real word")
            joke_start = input("Use proper words this time : ").split()
            prompt = [self.data['vocab'].word_to_idx[word] for word in joke_start]

        print(' '.join(joke_start), end=' ')

        while True:

            prompt_tensor = torch.tensor([prompt], device=self.device)
            
            y_pred, h = self(prompt_tensor, h)
            next_word_idx = np.argmax(F.softmax(y_pred[0], dim=0).detach().cpu().numpy())
            next_word = self.data['vocab'].idx_to_word[next_word_idx]
            print(next_word, end=' ')

            max_len = 50
            if next_word == '<period>' or next_word == '<pad>':
                print()
                return
            elif len(prompt) >= max_len:
                print("\n...Max output reached. Joke terminated.\n")
                return
            else:
                prompt.append(next_word_idx)

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
                outputs, _ = self(inputs, self.init_hidden(self.batch_size))
                pred_y.append(outputs.max(1)[1])
                true_y.append(labels)

            return torch.flatten(torch.stack(pred_y)), torch.flatten(torch.stack(true_y))
        elif self.DATASET == 1:
            self.eval()
            while True:
                self.predict_joke()



    def run(self):
        print('method running...')
        print('--start training...')
        self.embedding = nn.Embedding(self.vocab_input_size, self.input_dim).to(self.device)
        if self.DATASET == 1:
            self.fc = nn.Linear(self.hidden_dim1, self.vocab_input_size).to(self.device)

        self.train_data(self.data['train'])
        print('--start testing...')
        if self.DATASET == 0:
            pred_y, y_true = self.test(self.data['test'])
            return {'pred_y': pred_y, 'true_y': y_true}

        elif self.DATASET == 1:
            self.test(self.data['test'])
            return