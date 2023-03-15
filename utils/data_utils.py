import dgl
import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from utils.network_utils import get_graph_tensor_from_node_importance, cal_node_importance
from utils.pooling_params import pooling_sizes


def get_data_from_dataset(dataset, convert2tensor=True, node_importance=False):
    data = dgl.data.TUDataset(dataset)
    n_classes = data.num_classes
    graph_data = np.array([data[id] for id in range(len(data))], dtype=object)
    labels = np.array([g[1].numpy().tolist() for g in data])
    if convert2tensor:
        graph_tensors, x = Graphs2Tensor(graph_data, pooling_sizes[dataset])
        return graph_tensors, labels, n_classes, x
    elif node_importance:
        for i in range(len(graph_data)):
            print_progress(i, len(graph_data), prefix='adding node attr : ')
            graph_data[i][0].ndata['importance'] = torch.tensor(
                cal_node_importance(nx.Graph(dgl.to_networkx(graph_data[i][0]))), dtype=torch.float32)
        return graph_data, labels, n_classes
    else:
        return graph_data, labels, n_classes


def Graphs2Tensor(graphs, pooling_sizes):
    tensors = []
    x = []
    scaler = MinMaxScaler()
    for i, (graph, label) in enumerate(graphs):
        print_progress(i, len(graphs), prefix='converting graph to tensor: ')
        G = nx.Graph(dgl.to_networkx(graph))
        x.append(get_graph_tensor_from_node_importance(
            graph=G, measure='controllability', pooling_sizes=pooling_sizes
        ))
        tensors.append({
            'label': label
        })
    x = scaler.fit_transform(x)
    for t in range(len(tensors)):
        tensors[t]['graph_tensor'] = x[t]
    return np.array(tensors, dtype=object), x


def print_progress(now, total, length=20, prefix='progress:'):
    print('\r' + prefix + ' %.2f%%\t' % (now / total * 100), end='')
    print('[' + '>' * int(now / total * length) + '-' * int(length - now / total * length) + ']', end='')


def collate_tensors(samples):
    batch_size = len(samples)
    batch_graphs = [sample['graph_tensor'] for sample in samples]
    batch_labels = [sample['label'] for sample in samples]
    return torch.tensor(np.array(batch_graphs, dtype=np.float32)).squeeze().float().view(batch_size, -1), torch.tensor(
        np.array(batch_labels, dtype=np.float32)).squeeze().long()


def collate_graphs(samples):
    graphs, labels = map(list, zip(*samples))
    loop_graphs = [dgl.add_self_loop(graph) for graph in graphs]
    return dgl.batch(loop_graphs), torch.tensor(labels, dtype=torch.long)
