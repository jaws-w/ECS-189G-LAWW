'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        score_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        true_y = self.data['true_y'].cpu()
        pred_y = self.data['pred_y'].cpu()
        
        precision_list.append(precision_score(true_y, pred_y, average='weighted'))
        recall_list.append(recall_score(true_y, pred_y, average='weighted'))
        f1_list.append(f1_score(true_y, pred_y, average='weighted'))

        return accuracy_score(true_y, pred_y), \
                np.mean(score_list), np.std(score_list), \
                np.mean(precision_list), np.std(precision_list), \
                np.mean(recall_list), np.std(recall_list), \
                np.mean(f1_list), np.std(f1_list)
        