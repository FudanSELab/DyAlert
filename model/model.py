import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import HeteroConv, GraphConv
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)


class HGNN(torch.nn.Module):
    def __init__(self, out_dim, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU()
        self.num_layers = num_layers
        for i in range(num_layers):
            self.convs.append(HeteroConv({
            ('metric', 'correlation', 'metric'): GraphConv(in_channels=(-1, -1), out_channels=out_dim // 2 ** (num_layers - i - 1), aggr='add'),
            ('metric', 'cause', 'alert'): GraphConv(in_channels=(-1, -1), out_channels=out_dim // 2 ** (num_layers - i - 1), aggr='max')
            }, aggr='sum'))

    def forward(self, x, edge_index, edge_weight):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            x['alert'] = self.leaky_relu(x['alert'])
            x['metric'] = self.leaky_relu(x['metric'])
        return x['alert']


class MetricDGNNModel(torch.nn.Module):
    def __init__(self, lstm_dim=256, num_layers=2):
        super(MetricDGNNModel, self).__init__()
        self.gnn = HGNN(out_dim=lstm_dim, num_layers=num_layers)
        self.big_rnn = nn.GRU(input_size=lstm_dim, hidden_size=lstm_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.lin1 = nn.Sequential(nn.Linear(in_features=lstm_dim, out_features=lstm_dim // 2), nn.LeakyReLU(),
                                nn.Linear(in_features=lstm_dim // 2, out_features=lstm_dim // 4), nn.LeakyReLU())
        self.lin = nn.Linear(in_features=lstm_dim // 4, out_features=2)

        self.big_rnn_num_layers = 2
        self.big_rnn_hidden_size = lstm_dim
        self.rnn_hidden_size = lstm_dim

    # get spatial representation
    def forward(self, batched_data):
        batched_data = batched_data.to(device)
        node_embeddings = self.gnn(batched_data.x_dict, batched_data.edge_index_dict, batched_data.edge_weight_dict)
        return node_embeddings

    # get spatio-temporal representation and conduct link prediction
    def entire_forward(self, gnn_node_embeddings, nodes, data, nodes_rel, rnn_batch_size, h0, step=None):
        gnn_node_embeddings_dataset = FlexibleRnnData(gnn_node_embeddings)
        gnn_nodes_dataloader = DataLoader(gnn_node_embeddings_dataset, batch_size=rnn_batch_size,
                                          collate_fn=collate_fn)
        node_embeddings = []
        node_lens = []
        for d in gnn_nodes_dataloader:
            d = d.to(device)
            outputs, hn = self.big_rnn(d, h0)
            outputs, out_len = pad_packed_sequence(outputs, batch_first=True, padding_value=-1)
            node_embeddings.append(outputs)
            node_lens.append(out_len)

        # conduct link prediction
        predicted_y = torch.tensor([]).to(device)
        actual_y = torch.tensor([]).to(device)
        names_pair = []
        node_idx = [0] * len(nodes)
        for idx in range(len(data['alert'].names)):
            # only predict links among alerts in the last step snapshot
            if step is not None and idx < len(data['alert'].names) - step:
                for i in range(len(data['alert'].names[idx])):
                    node_idx[nodes_rel[data['alert'].names[idx][i]]] += 1
                continue
            s = data['alert'].ptr[idx]
            compare_pair = []
            for i in range(len(data['alert'].names[idx])):
                if data['alert'].is_new[idx][i] == 0:
                    continue
                for j in range(len(data['alert'].names[idx])):
                    if i == j or ((i, j) in compare_pair or (j, i) in compare_pair):
                        continue
                    compare_pair.append((i, j))
                    name_i = data['alert'].names[idx][i]
                    i_idx = nodes_rel[name_i]
                    name_j = data['alert'].names[idx][j]
                    j_idx = nodes_rel[name_j]
                    names_pair.append((name_i, name_j))
                    node_embeddings_i = node_embeddings[i_idx // rnn_batch_size][i_idx % rnn_batch_size, node_idx[i_idx], :].to(device)
                    node_embeddings_j = node_embeddings[j_idx // rnn_batch_size][j_idx % rnn_batch_size, node_idx[j_idx], :].to(device)

                    predicted_ans = self.lin1((node_embeddings_i - node_embeddings_j) ** 2)
                    predicted_ans = self.lin(predicted_ans)
                    predicted_y = torch.cat((predicted_y, predicted_ans.reshape([1, -1])), 0)
                    actual_y = torch.cat((actual_y, data['alert'].link_edges[s + i][j].reshape([1])), 0).to(device)
            for i in range(len(data['alert'].names[idx])):
                node_idx[nodes_rel[data['alert'].names[idx][i]]] += 1
        return predicted_y, actual_y, names_pair


class FlexibleRnnData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(data):
    lengths = [d.shape[0] for d in data]
    data = pad_sequence(data, batch_first=True, padding_value=-1)
    data = pack_padded_sequence(data, lengths, batch_first=True, enforce_sorted=False)
    return data



