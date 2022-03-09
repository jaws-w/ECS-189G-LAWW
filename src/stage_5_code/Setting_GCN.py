from src.base_class.setting import setting

# from base_class.setting import setting
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import torch

class Setting_GCN(setting):
    dataset_test = None
    dataset_train = None

    def load_presplit_data(self, dataset_test, dataset_train):
        self.dataset_test = dataset_test
        self.dataset_train = dataset_train

    def load_run_save_evaluate(self):
        score_list = []
        precision_list = []
        recall_list = []
        f1_list = []

        # load data
        self.method.data = self.dataset.load()
        # self.method.batch_size = self.dataset.batch_size
        # self.method.vocab_input_size = self.dataset.vocab_size
        # self.method.out_size = self.dataset.label_size

        # run module
        learned_result = self.method.run()
        
        # save result
        self.result.data = learned_result

        self.evaluate.data = learned_result
        score_list.append(self.evaluate.evaluate())

        true_y = self.result.data['true_y'].cpu()
        pred_y = self.result.data['pred_y'].cpu()
        precision_list.append(precision_score(true_y, pred_y, average='weighted'))
        recall_list.append(recall_score(true_y, pred_y, average='weighted'))
        f1_list.append(f1_score(true_y, pred_y, average='weighted'))

        return accuracy_score(true_y, pred_y),\
                np.mean(score_list), np.std(score_list), \
                np.mean(precision_list), np.std(precision_list), \
                np.mean(recall_list), np.std(recall_list), \
                np.mean(f1_list), np.std(f1_list)
