import os

import torch
from heterogeneous_graph_data_construction import HeterogeneousGraphDataset
from model.customized_dataloader import DataLoader
from model.sliding_dataset import SlidingDataset
from model.model import MetricDGNNModel
from tqdm import tqdm
import logging
import time
import pandas as pd
from sklearn import metrics

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test_for_test_dataset(slide_window,
                          test_data_split=(0.7, 0.8), model_path='./models',
                          correlation_method='pearsonr', metric_length='all', gnn_num_layers=2,
                          metric_correlation_threshold=None,
                          model_file='model.pth'):
    # log
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(
        f'./logs/test_slideWindow_{slide_window[0]},{slide_window[1]}_testDataSplit_{test_data_split}_gnn_num_layers_{gnn_num_layers}_metricCorrelationThreshold_{metric_correlation_threshold}_log.txt')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    dataset = HeterogeneousGraphDataset('.', metric_length=metric_length,
                                        correlation_method=correlation_method,
                                        metric_correlation_threshold=metric_correlation_threshold)
    # init dataloader
    dataloader_test = DataLoader(SlidingDataset(dataset[int(len(dataset) * test_data_split[0]):int(len(dataset) * test_data_split[1])], window=slide_window[0], step=slide_window[1]))
    logger.info('--------------------------')
    logger.info(
        'test snapshot num: ' + str(int(len(dataset) * test_data_split[1]) - int(len(dataset) * test_data_split[0])))
    logger.info(f'model name: {model_file}')
    logger.info(f'eval_data_split: {test_data_split}')
    logger.info(f'slide_window: {slide_window}')

    model = MetricDGNNModel(lstm_dim=256, num_layers=gnn_num_layers).to(device)
    model.load_state_dict(torch.load(model_path + '/' + model_file))

    # evaluation
    model.eval()
    correct = 0
    total = 0
    result_dict = {'alert_1': [], 'alert_2': [], 'ground_truth': [], 'prediction': []}
    TP = 0  # True Positive
    FN = 0  # False Negative
    FP = 0  # False Positive
    TN = 0  # True Negative
    start_time = time.time()
    with torch.no_grad():
        for data in tqdm(dataloader_test):
            nodes = []
            entire_rnn_data = []
            node_embedding_output = model.forward(data)
            nodes_rel = {}

            idx = 0

            for i, batch_name in enumerate(data['alert'].names):
                for j, name in enumerate(batch_name):
                    if i == 0 or data['alert'].is_new[i][j] == 1:
                        # new alert
                        nodes.append(name)
                        entire_rnn_data.append(torch.tensor([]).to(device))
                        nodes_rel[name] = len(nodes) - 1
                    entire_rnn_data[nodes_rel[name]] = torch.cat((entire_rnn_data[nodes_rel[name]], node_embedding_output[idx].reshape([1, -1])), 0)
                    idx += 1
            h0 = torch.zeros((model.big_rnn_num_layers, len(nodes), model.big_rnn_hidden_size)).to(device)
            predicted_y, actual_y, names_pair = model.entire_forward(entire_rnn_data, nodes, data,
                                                                                 rnn_batch_size=16, h0=h0,
                                                                                 step=slide_window[1],
                                                                                 nodes_rel=nodes_rel)

            prediction = torch.argmax(predicted_y, 1)
            total += actual_y.shape[0]
            correct += (prediction == actual_y).sum().float()
            for i, pair in enumerate(names_pair):
                result_dict['alert_1'].append(pair[0])
                result_dict['alert_2'].append(pair[1])
                result_dict['ground_truth'].append(actual_y[i].cpu().numpy())
                result_dict['prediction'].append(prediction[i].cpu().numpy())

            for i in range(prediction.shape[0]):
                if prediction[i] == 1:
                    if actual_y[i] == 1:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if actual_y[i] == 1:
                        FN += 1
                    else:
                        TN += 1
    precision = TP / (TP + FP) if TP + FP != 0 else None
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall) if precision is not None else None
    logger.info(
        f'Total: {total}, EvalAccuracy: {(correct / total).cpu():.2f}, Precision: {precision}, Recall: {recall}, F1-score: {f1_score}, Time: {time.time() - start_time:.2f}s')
    # output classification result
    os.makedirs('./results', exist_ok=True)
    result_df = pd.DataFrame(result_dict)
    result_df = result_df.set_index(['alert_1', 'alert_2'])
    result_df.to_csv(
        f'./results/slideWindow_{slide_window[0]},{slide_window[1]}_evalDataSplit_{test_data_split}_gnn_num_layers_{gnn_num_layers}_metricCorrelationThreshold_{metric_correlation_threshold}_result.csv')


# find parent node
def find(item, alert_dict):
    stack = []
    while item != alert_dict[item]:
        stack.append(item)
        item = alert_dict[item]
    while len(stack) > 1:
        alert_dict[stack.pop()] = item
    return item


# merge two set
def union(a, b, tree_dict, alert_dict):
    fa = find(a, alert_dict)
    fb = find(b, alert_dict)
    if fa!= fb:
        if tree_dict[fa] >= tree_dict[fb]:
            alert_dict[fb] = fa
            if tree_dict[fa] == tree_dict[fb]:
                tree_dict[fa] += 1
            del tree_dict[fb]
        else:
            alert_dict[fa] = fb
            del tree_dict[fa]


def get_cluster_results(file_name, type='ground_truth'):
    results = pd.read_csv(file_name)
    alert_dict = {}
    tree_dict = {}

    for _, row in results.iterrows():
        alert_1 = row['alert_1'].split('#')
        alert_2 = row['alert_2'].split('#')
        for alert in alert_1 + alert_2:
            if alert.split('*')[0].split('_')[0] not in alert_dict.keys():
                alert_dict[alert.split('*')[0].split('_')[0]] = alert.split('*')[0].split('_')[0]
                tree_dict[alert.split('*')[0].split('_')[0]] = 1
    # merge alerts with prediction
    for _, row in results.iterrows():
        alert_1 = row['alert_1'].split('#')
        alert_2 = row['alert_2'].split('#')
        if row[type] == 1:
            for alert_src in alert_1:
                for alert_dst in alert_2:
                    alert_src = alert_src.split('*')[0].split('_')[0]
                    alert_dst = alert_dst.split('*')[0].split('_')[0]
                    union(alert_src, alert_dst, tree_dict, alert_dict)
    return alert_dict, tree_dict


def get_cluster_measures(file_name):
    ground_truth_results = get_cluster_results(file_name, type='ground_truth')
    predict_results = get_cluster_results(file_name, type='prediction')
    print(f'ground_truth num: {len(ground_truth_results[1].keys())}')
    idx = 0
    num_dict = {}
    ground_truth_cluster_ids = []
    for alert in sorted(ground_truth_results[0].keys()):
        tmp = find(alert, ground_truth_results[0])
        if tmp not in num_dict.keys():
            num_dict[find(alert, ground_truth_results[0])] = idx
            idx += 1
        ground_truth_cluster_ids.append(num_dict[tmp])
    print(f'ground_truth_cluster_ids: {ground_truth_cluster_ids}')

    print(f'predict num: {len(predict_results[1].keys())}')
    idx = 0
    num_dict = {}
    predict_cluster_ids = []
    for alert in sorted(predict_results[0].keys()):
        tmp = find(alert, predict_results[0])
        if tmp not in num_dict.keys():
            num_dict[find(alert, predict_results[0])] = idx
            idx += 1
        predict_cluster_ids.append(num_dict[tmp])
    print(f'predict_cluster_ids: {predict_cluster_ids}')
    print(f'AMI Score: {metrics.cluster.adjusted_mutual_info_score(ground_truth_cluster_ids, predict_cluster_ids)}')


if __name__ == '__main__':
    model = 'epochs_50_loss_focal loss_lossParam_None,1.5_slideWindow_10,8_gnn_num_layers_2_metricCorrelationThreshold_0.6_model.pth'
    test_for_test_dataset(slide_window=(10, 8),
                          model_path='./models/pearson/data_split0.8/metricLength_30',
                          model_file=model, correlation_method='pearson',
                          metric_length=30, gnn_num_layers=2, metric_correlation_threshold=0.6, test_data_split=(0.8, 1.0))

    print('--------Our approach-------')
    get_cluster_measures('results/slideWindow_10,8_evalDataSplit_(0.8, 1.0)_gnn_num_layers_2_metricCorrelationThreshold_0.6_result.csv')
