from src.base_class.setting import setting
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

        return np.mean(score_list), np.std(score_list)
