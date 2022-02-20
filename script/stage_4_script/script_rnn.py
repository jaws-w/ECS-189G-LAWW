from src.stage_4_code.Dataset_Loader import Dataset_Loader
from src.stage_4_code.Method_RNN import Method_RNN
from src.stage_4_code.Result_Saver import Result_Saver
from src.stage_4_code.Setting_RNN import Setting_RNN
from src.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy

if 1:
    # CLASSIFICATION: 0, GENERATION: 1
    DATASET = 0

    if DATASET == 0:
        data_obj = Dataset_Loader('CLASSIFICATION', '')
        data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification'
        # data_obj.dataset_source_file_name = ''
    elif DATASET == 1:
        data_obj = Dataset_Loader('GENERATION', '')
        data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation'
        data_obj.dataset_source_file_name = 'ORL'

    data_obj.load()
    print(data_obj.max_length)
    # method_obj = Method_RNN('RNN', '', DATASET, data_obj)
    

    # result_obj = Result_Saver('saver', '')
    # result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_'
    # # result_obj.result_destination_file_name = 'prediction_result'
    # result_obj.result_destination_file_name = 'prediction_result_ORL'
    # # result_obj.result_destination_file_name = 'prediction_result_CIFAR'

    # setting_obj = Setting_RNN('CNN Setting', '')

    # evaluate_obj = Evaluate_Accuracy('accuracy', '')


    # # ---- running section ---------------------------------
    # print('************ Start ************')
    # setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    # setting_obj.print_setup_summary()
    # accuracy, mean_score, std_score, avg_precision, std_precision, avg_recall, std_recall, avg_f1, std_f1 = \
    #     setting_obj.load_run_save_evaluate()
    # # accuracy = setting_obj.load_run_save_evaluate()
    # print('************ Overall Performance ************')
    # print('CNN Accuracy: ' + str(accuracy))
    # print('CNN Precision: ' + str(avg_precision) + ' +/- ' + str(std_precision))
    # print('CNN Recall: ' + str(avg_recall) + ' +/- ' + str(std_recall))
    # print('CNN F1: ' + str(avg_f1) + ' +/- ' + str(std_f1))
    # print('************ Finish ************')
    # # ------------------------------------------------------