import numpy as np
import torch
from heterogeneous_graph_data_construction import HeterogeneousGraphDataset
from model.customized_dataloader import DataLoader
from model.sliding_dataset import SlidingDataset
from torch.optim import lr_scheduler
from tqdm import tqdm
from model.focal_loss import FocalLoss
from model.model import MetricDGNNModel
import logging
import time
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(epochs, lr, loss_method, loss_param=None, slide_window=(10, 8),
          metric_length=0, correlation_method='pearson', data_split=0.8,
          gnn_num_layers=2, trained_model=None, metric_correlation_threshold=None):
    # log
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    dir = f'./logs/{correlation_method}/data_split{data_split}/metricLength_{metric_length}'
    os.makedirs(dir, exist_ok=True)
    handler = logging.FileHandler(
        f'{dir}/train_epochs_{epochs}_loss_{loss_method}_lossParam_{loss_param[0]},{loss_param[1]}_slideWindow_{slide_window[0]},{slide_window[1]}_gnn_num_layers_{gnn_num_layers}_metricCorrelationThreshold_{metric_correlation_threshold}_log.txt')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    dataset = HeterogeneousGraphDataset('.', metric_length=metric_length,
                                        correlation_method=correlation_method,
                                        metric_correlation_threshold=metric_correlation_threshold)
    # init dataloader
    dataloader_train = DataLoader(SlidingDataset(dataset[:int(len(dataset) * data_split)], window=slide_window[0], step=slide_window[1]), shuffle=True)

    logger.info('--------------------------')
    logger.info('train dataloader num: ' + str(int(len(dataset) * data_split)))
    logger.info('eval dataloader num: ' + str(int(len(dataset)) - int(len(dataset) * data_split)))
    logger.info(f'slide_window: {slide_window}')
    logger.info(f'learning_rate: {lr}')
    logger.info(f'epochs: {epochs}')
    logger.info(f'loss: {loss_method}')
    logger.info(f'metric_length: {metric_length}')
    logger.info(f'correlation_method: {correlation_method}')
    logger.info(f'gnn_num_layer: {gnn_num_layers}')
    logger.info(f'metric_correlation_threshold: {metric_correlation_threshold}')
    if loss_param is not None:
        if loss_method == 'focal loss':
            logger.info(f'alpha: {loss_param[0]}, gamma: {loss_param[1]}')
        else:
            logger.info(f'weight of 0: {loss_param[0]}, weight of 1: {loss_param[1]}')
    logger.info('--------------------------')
    model = MetricDGNNModel(lstm_dim=256, num_layers=gnn_num_layers).to(device)

    if trained_model is not None:
        logger.info('Continue training!!!')
        logger.info(f'Model Name: {trained_model}')
        model_path = f'./models/{correlation_method}/data_split{data_split}/metricLength_{metric_length[0]},{metric_length[1]}'
        model.load_state_dict(torch.load(model_path + '/' + trained_model))
        model_params = trained_model.split('_')
        base_epoch = int(model_params[model_params.index('epochs') + 1])
    else:
        base_epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if loss_method == 'focal loss':
        # focal loss
        loss_fn = FocalLoss(alpha=loss_param[0], gamma=loss_param[1]).to(device)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)  # 学习率调度器

    for epoch in tqdm(range(base_epoch, epochs)):
        model.train()

        epoch_loss = 0
        epoch_time = time.time()
        total = 0
        for data in dataloader_train:
            optimizer.zero_grad()
            # if there is only one alert on the snapshot, no need to predict links.
            continue_flag = True
            for i, batch_name in enumerate(data['alert'].names):
                if slide_window[0] - slide_window[1] <= i:
                    if len(batch_name) > 1:
                        continue_flag = False
            if continue_flag:
                continue

            nodes = []
            entire_rnn_data = []
            nodes_rel = {}
            node_embedding_output = model.forward(data)
            idx = 0

            for i, batch_name in enumerate(data['alert'].names):
                for j, name in enumerate(batch_name):
                    if i == 0 or data['alert'].is_new[i][j] == 1:
                        # new alert
                        nodes.append(name)
                        entire_rnn_data.append(torch.tensor([]).to(device))
                        nodes_rel[name] = len(nodes) - 1
                    entire_rnn_data[nodes_rel[name]] = torch.cat((entire_rnn_data[nodes_rel[name]],node_embedding_output[idx].reshape([1, -1])), 0)
                    idx += 1
            h0 = torch.zeros((model.big_rnn_num_layers, len(nodes), model.big_rnn_hidden_size)).to(device)
            predicted_y, actual_y, _ = model.entire_forward(entire_rnn_data, nodes, data, nodes_rel=nodes_rel, rnn_batch_size=16,
                                                                        h0=h0, step=slide_window[1])

            total += actual_y.shape[0]
            loss = loss_fn(predicted_y.float(), actual_y.long())
            epoch_loss += loss
            loss.backward()
            optimizer.step()
        scheduler.step(epoch_loss)
        logger.info(
            f'Epoch: {epoch + 1:03d}, LearningRate: {optimizer.param_groups[0]["lr"]}, EpochLoss: {epoch_loss:.4f}, EpochTime: {time.time() - epoch_time:.2f}s')
        if (epoch + 1) % 50 == 0 or epoch + 1 == epochs:
            # save model
            dir = f'./models/{correlation_method}/data_split{data_split}/metricLength_{metric_length}'
            os.makedirs(dir, exist_ok=True)
            torch.save(model.state_dict(),
                       f'{dir}/epochs_{epoch + 1}_loss_{loss_method}_lossParam_{loss_param[0]},{loss_param[1]}_slideWindow_{slide_window[0]},{slide_window[1]}_gnn_num_layers_{gnn_num_layers}_metricCorrelationThreshold_{metric_correlation_threshold}_model.pth')


if __name__ == '__main__':
    train(epochs=50, lr=0.0001, loss_method='focal loss',
          correlation_method='pearson', loss_param=(None, 1.5), slide_window=(10, 8),
          metric_length=30, data_split=0.8, gnn_num_layers=2,
          trained_model=None, metric_correlation_threshold=0.6)