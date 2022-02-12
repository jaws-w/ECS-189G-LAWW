from src.stage_3_code.Dataset_Loader import Dataset_Loader
from src.stage_3_code.Method_CNN import Method_CNN
from src.stage_3_code.Result_Saver import Result_Saver
from src.stage_3_code.Setting_CNN import Setting_CNN
from src.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy

CIFAR_classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if 1:

    # data_obj = Dataset_Loader('MNIST', '')
    # data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    # data_obj.dataset_source_file_name = 'MNIST'

    data_obj = Dataset_Loader('ORL', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = 'ORL'

    # data_obj = Dataset_Loader('CIFAR', '')
    # data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    # data_obj.dataset_source_file_name = 'CIFAR'

    method_obj = Method_CNN('CNN', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_'
    # result_obj.result_destination_file_name = 'prediction_result'
    result_obj.result_destination_file_name = 'prediction_result_ORL'
    # result_obj.result_destination_file_name = 'prediction_result_CIFAR'

    setting_obj = Setting_CNN('CNN Setting', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')


    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    # mean_score, std_score, avg_precision, std_precision, avg_recall, std_recall, avg_f1, std_f1 = \
        # setting_obj.load_run_save_evaluate()
    accuracy = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(accuracy))
    # print('MLP Precision: ' + str(avg_precision) + ' +/- ' + str(std_precision))
    # print('MLP Recall: ' + str(avg_recall) + ' +/- ' + str(std_recall))
    # print('MLP F1: ' + str(avg_f1) + ' +/- ' + str(std_f1))
    print('************ Finish ************')
    # ------------------------------------------------------