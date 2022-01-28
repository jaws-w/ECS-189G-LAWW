from src.stage_2_code.Dataset_Loader import Dataset_Loader
from src.stage_2_code.Method_MLP import Method_MLP
from src.stage_2_code.Result_Saver import Result_Saver
from src.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from src.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from src.stage_2_code.Setting import Setting
from src.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    train_data_obj = Dataset_Loader('train', '')
    train_data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    train_data_obj.dataset_source_file_name = 'train.csv'

    test_data_obj = Dataset_Loader('test', '')
    test_data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    test_data_obj.dataset_source_file_name = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting('Pre-split setting', '')
    setting_obj.load_presplit_data(test_data_obj, train_data_obj)
    # setting_obj = Setting_KFold_CV('k fold cross validation', '')
    # setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(train_data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score, avg_precision, std_precision, avg_recall, std_recall, avg_f1, std_f1 = \
        setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('MLP Precision: ' + str(avg_precision) + ' +/- ' + str(std_precision))
    print('MLP Recall: ' + str(avg_recall) + ' +/- ' + str(std_recall))
    print('MLP F1: ' + str(avg_f1) + ' +/- ' + str(std_f1))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    