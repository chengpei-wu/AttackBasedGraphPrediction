import dgl
import torch

from utils.attack_utils import *
from utils.network_utils import get_graph_tensor_from_node_importance, cal_node_importance
from utils.params import pooling_sizes, attack_measures


def get_data_from_dataset(dataset, convert2tensor=True, tensor_from='node', node_importance=False):
    data = dgl.data.TUDataset(dataset)
    n_classes = data.num_classes
    graph_data = np.array([data[id] for id in range(len(data))], dtype=object)
    labels = np.array([g[1].numpy().tolist() for g in data])
    if convert2tensor:
        # return the representation tensor of each graph instead of the graph structure data
        # tensor from two aspects: 1) curve under attacks: 'graph', 2) node importance through pplp: 'node'
        graph_tensors, x = Graphs2Tensor(graph_data, pooling_sizes[dataset], tensor_from=tensor_from)
        return graph_tensors, labels, n_classes, x
    elif node_importance:
        for i in range(len(graph_data)):
            print_progress(i, len(graph_data), prefix='adding node attr : ')
            graph_data[i][0].ndata['importance'] = torch.tensor(
                cal_node_importance(nx.Graph(dgl.to_networkx(graph_data[i][0])), measure='all'), dtype=torch.float32)
        return graph_data, labels, n_classes
    else:
        return graph_data, labels, n_classes


def Graphs2Tensor(graphs, pooling_sizes, tensor_from='node'):
    tensors = []
    x = []
    for i, (graph, label) in enumerate(graphs):
        print_progress(i, len(graphs), prefix='converting graph to tensor: ')
        G = nx.Graph(dgl.to_networkx(graph))
        if tensor_from == 'node':
            x.append(get_graph_tensor_from_node_importance(
                graph=G, measure='all', pooling_sizes=pooling_sizes
            ))
        elif tensor_from == 'graph':
            x.append(get_graph_tensor_from_graph_attack(
                graph=G, measures=attack_measures, fixed='sample', sample_size=5
            ))
        else:
            x.append(
                [
                    cal_controllability(G),
                    cal_connectivity(G),
                    cal_average_degree(G),
                    cal_betweenness(G),
                    cal_clustering_coefficient(G),
                    cal_communication_ability(G),
                    cal_adj_spectral_radius(G),
                    # cal_adj_spectral_gap(G),
                    cal_natural_connectivity(G),
                    cal_algebraic_connectivity(G)
                ]
            )
        tensors.append({
            'label': label
        })
    x = np.array(x)
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


def dataset_visualization(dataset):
    data = dgl.data.TUDataset(dataset)
    graph_data = np.array([data[id] for id in range(len(data))], dtype=object)
    circles1, circles2 = [], []
    diameter1, diameter2 = [], []
    for graph, label in graph_data:
        nx_g = nx.Graph(graph.to_networkx())
        if label.numpy().tolist() == [0]:
            circles1.append(len(nx.cycle_basis(nx_g)))
            diameter1.append(nx.diameter(nx_g))
            # print(circles1, diameter1)
        else:
            circles2.append(len(nx.cycle_basis(nx_g)))
            diameter2.append(nx.diameter(nx_g))
            # print(circles2, diameter2)
        # nx.draw(nx_g)
        # plt.show()
    print(circles1)
    print(circles2)
    print(diameter1)
    print(diameter2)

# dataset_visualization('MUTAG')
