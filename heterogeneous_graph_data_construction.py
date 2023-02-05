from torch_geometric.data import HeteroData, Dataset
from model.bert import *
import multiprocessing
import os
import json
import os.path as osp
from tqdm import tqdm
import numpy as np


class HeterogeneousGraphDataset(Dataset):
    def __init__(self, root, metric_length=0, metric_correlation_threshold=None,
                 correlation_method='pearson', transform=None, pre_transform=None, pre_filter=None):
        self.metric_length = metric_length
        self.correlation_method = correlation_method
        self.metric_correlation_threshold = metric_correlation_threshold
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        file_dict = {}
        # list all snapshots json files and sort them
        # each file contains 50 snapshots in our experiment
        for file in os.listdir(self.raw_dir):
            if file.endswith('.json') and file.startswith('snapshots'):
                file_dict[int(file[file.index('-') + 1: -5])] = file
        file_list = [file_dict[k] for k in sorted(file_dict.keys())]
        return file_list

    @property
    def raw_dir(self) -> str:
        dir = self.root + '/data/metric_' + str(self.metric_length)
        return dir

    @property
    def processed_dir(self) -> str:
        dir = self.root + '/processed'
        if self.metric_correlation_threshold is not None:
            dir += f'/threshold_{self.metric_correlation_threshold}'
        os.makedirs(dir, exist_ok=True)
        return dir

    @property
    def processed_file_names(self):
        file_list = []
        for file in os.listdir(self.processed_dir):
            if file in ['pre_filter.pt', 'pre_transform.pt']:
                continue
            if file.startswith(f'snapshot_'):
                file_list.append(file)
        return sorted(file_list)

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int):
        data = torch.load(
            osp.join(self.processed_dir, f'snapshot_{idx}.pt'))
        return data

    def sub_process(self, file, idx, bert):
        with open(file, 'r') as f:
            snapshots = json.load(f)
        print(len(snapshots))
        for it, snapshot in tqdm(enumerate(snapshots)):
            graph_data = HeteroData()
            # get alert embedding
            graph_data['alert'].x = bert.get_batch_sentence_embedding(snapshot['messages'], 128)
            graph_data['alert'].names = snapshot['alert_names']
            graph_data['alert'].is_new = snapshot['alert_is_new']

            # alert_link_alert_edges
            y_edges = np.eye(graph_data['alert'].x.shape[0])
            for i in range(len(snapshot['alert_link_alert_edges'][0])):
                y_edges[snapshot['alert_link_alert_edges'][0][i]][snapshot['alert_link_alert_edges'][1][i]] = 1
            graph_data['alert'].link_edges = torch.tensor(y_edges, dtype=torch.long)

            # metric
            graph_data['metric'].x = torch.tensor(np.asarray(snapshot['metric_x']), dtype=torch.float)
            # normalization
            graph_data['metric'].x = torch.nn.functional.normalize(graph_data['metric'].x)

            # metric_correlation_metric_edges
            graph_data['metric', 'correlation', 'metric'].edge_weight = torch.tensor(
                np.asarray(snapshot['metric_correlation_metric_edges_weight']),
                dtype=torch.float)
            if self.correlation_method == 'abs_pearsonr':
                graph_data['metric', 'correlation', 'metric'].edge_weight = torch.abs(
                    graph_data['metric', 'correlation', 'metric'].edge_weight)

            edge_index = snapshot['metric_correlation_metric_edges']
            graph_data['metric', 'correlation', 'metric'].edge_index = torch.tensor(
                np.asarray(edge_index), dtype=torch.long)
            # metric_cause_alert_edge
            graph_data['metric', 'cause', 'alert'].edge_index = torch.tensor(
                np.asarray([snapshot['alert_from_metric_edges'][1], snapshot['alert_from_metric_edges'][0]]),
                dtype=torch.long)
            edge_weight = [1] * len(graph_data['metric', 'cause', 'alert'].edge_index[0])
            graph_data['metric', 'cause', 'alert'].edge_weight = torch.tensor(np.asarray(edge_weight), dtype=torch.float)
            if self.pre_filter is not None and not self.pre_filter(graph_data):
                return

            if self.pre_transform is not None:
                graph_data = self.pre_transform(graph_data)
            torch.save(graph_data, osp.join(self.processed_dir,
                                            f'snapshot_{idx * 50 + it}.pt'))

    def process(self):
        pool = multiprocessing.Pool(processes=5)
        bert = Bert(tokenizer_pretrained='hfl/chinese-bert-wwm-ext', model_pretrained='hfl/chinese-bert-wwm-ext')
        idx = 0

        for file in tqdm(self.raw_paths):
            pool.apply_async(self.sub_process, (file, idx, bert))
            idx += 1
        pool.close()
        pool.join()

    def process_data_with_threshold(self, metric_correlation_threshold):
        pool = multiprocessing.Pool(processes=3)
        dir = f'{self.processed_dir}/threshold_{metric_correlation_threshold}'
        os.makedirs(dir, exist_ok=True)
        for i in tqdm(range(self.len())):
            data = self.get(i)
            pool.apply_async(self.mask_metric_edges_based_on_threshold, (data, metric_correlation_threshold, dir, i))
        pool.close()
        pool.join()

    def mask_metric_edges_based_on_threshold(self, data, metric_correlation_threshold, dir, idx):
        edge_weight_array = data['metric', 'correlation', 'metric'].edge_weight.cpu().numpy()
        edge_index_array = data['metric', 'correlation', 'metric'].edge_index.cpu().numpy()
        delete_ids = []
        for i in range(edge_weight_array.shape[0]):
            if -metric_correlation_threshold < edge_weight_array[i] < metric_correlation_threshold:
                delete_ids.append(i)
        edge_weight_array = np.delete(edge_weight_array, delete_ids, 0)
        edge_index_array = np.delete(edge_index_array, delete_ids, 1)
        data['metric', 'correlation', 'metric'].edge_weight = torch.tensor(edge_weight_array, dtype=torch.float)
        data['metric', 'correlation', 'metric'].edge_index = torch.tensor(edge_index_array, dtype=torch.long)
        torch.save(data, osp.join(dir, f'snapshot_{idx}.pt'))
        print(f'snapshot_{idx}.pt')


if __name__ == '__main__':
    # You need to init data with param 'metric_correlation_threshold=None'
    dataset = HeterogeneousGraphDataset('.', correlation_method='pearson', metric_length=30,
                                        metric_correlation_threshold=None)
    print(dataset)
    print(dataset[0])
    # # if you need to filter correlation edges among metrics, you need to run this function.
    # dataset.process_data_with_threshold(metric_correlation_threshold=0.6)
    #
    # # Then you can use this function with metric_correlation_threshold=threshold to get the dataset you want
    # dataset = HeterogeneousGraphDataset('.', correlation_method='pearson', metric_length=30,
    #                                     metric_correlation_threshold=0.6)
    # print(dataset)
    # print(dataset[0])
