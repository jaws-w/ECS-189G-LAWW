from stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from stage_5_code.Method_GCN import Method_GCN
from stage_5_code.Result_Saver import Result_Saver
from stage_5_code.Setting_GCN import Setting_GCN
from stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy

# from src.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
# from src.stage_5_code.Method_GCN import Method_GCN
# from src.stage_5_code.Result_Saver import Result_Saver
# from src.stage_5_code.Setting_GCN import Setting_GCN
# from src.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy

if 1:
    # cora: 0, citeseer: 1, pubmed: 2, cora-small: 3
    DATASET = 3

    DATA_FOLDER = '../../data/stage_5_data/'

    if DATASET == 3:
        data_obj = Dataset_Loader(dName='cora')
        data_obj.dataset_source_folder_path = DATA_FOLDER + 'cora'
    elif DATASET == 1:
        data_obj = Dataset_Loader(dName='citeseer')
        data_obj.dataset_source_folder_path = DATA_FOLDER + 'citeseer'
    elif DATASET == 2:
        data_obj = Dataset_Loader(dName='pubmed')
        data_obj.dataset_source_folder_path = DATA_FOLDER + 'pubmed'
    elif DATASET == 3:
        data_obj = Dataset_Loader(dName='cora-small')
        data_obj.dataset_source_folder_path = DATA_FOLDER + 'cora'
        

    # init objects to run the RNN model.
    method_obj = Method_GCN('GCN', '', DATASET)
    setting_obj = Setting_GCN('GCN Setting', '')
    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    result_obj = Result_Saver('saver', '')

    # # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    accuracy, mean_score, std_score, avg_precision, std_precision, avg_recall, std_recall, avg_f1, std_f1 = \
        setting_obj.load_run_save_evaluate()
    # accuracy = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN Accuracy: ' + str(accuracy))
    print('CNN Precision: ' + str(avg_precision) + ' +/- ' + str(std_precision))
    print('CNN Recall: ' + str(avg_recall) + ' +/- ' + str(std_recall))
    print('CNN F1: ' + str(avg_f1) + ' +/- ' + str(std_f1))
    print('************ Finish ************')
    # # ------------------------------------------------------