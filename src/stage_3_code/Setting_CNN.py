'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from src.base_class.setting import setting
import numpy as np

class Setting_CNN(setting):
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()

        #X_train, X_test, y_train, y_test = loaded_data['X'], loaded_data['y'], test_size = 0.33)

        # run MethodModule
        self.method.data = loaded_data
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result

        true_y = self.result.data['true_y']
        pred_y = self.result.data['pred_y']
        
        # return self.evaluate.evaluate(), None
        return self.evaluate.evaluate()

        