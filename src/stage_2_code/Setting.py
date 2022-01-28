from src.base_class.setting import setting
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

class Setting(setting):
    dataset_test = None
    dataset_train = None

    def load_presplit_data(self, dataset_test, dataset_train):
        self.dataset_test = dataset_test
        self.dataset_train = dataset_train

    def load_run_save_evaluate(self):
        loaded_train_data = self.dataset_train.load()

        loaded_test_data = self.dataset_test.load()
        # print(loaded_test_data)
        # print(loaded_train_data)

        score_list = []
        precision_list = []
        recall_list = []
        f1_list = []

        


        # run MethodModule
        # self.method.data = {'train': {'X': loaded_data['X'], 'y': loaded_data['y']}, 'test': {'X': X_test, 'y': y_test}}
        self.method.data = {'train': {'X': loaded_train_data['X'], 'y': loaded_train_data['y']}, 'test': {'X': loaded_test_data['X'], 'y': loaded_test_data['y']}}
        learned_result = self.method.run()
        
        # save raw ResultModule
        self.result.data = learned_result
        #self.result.fold_count = fold_count
        self.result.save()

        self.evaluate.data = learned_result
        score_list.append(self.evaluate.evaluate())

        true_y = self.result.data['true_y']
        pred_y = self.result.data['pred_y']
        precision_list.append(precision_score(true_y, pred_y, average='weighted'))
        recall_list.append(recall_score(true_y, pred_y, average='weighted'))
        f1_list.append(f1_score(true_y, pred_y, average='weighted'))

        return np.mean(score_list), np.std(score_list), \
               np.mean(precision_list), np.std(precision_list), \
               np.mean(recall_list), np.std(recall_list), \
               np.mean(f1_list), np.std(f1_list)
