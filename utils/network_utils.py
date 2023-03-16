import networkx
import networkx as nx
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.attack_utils import cal_controllability, cal_connectivity
from utils.pyramid_pooling_utils import pyramid_pooling


def get_graph_tensor_from_node_importance(graph: nx.Graph, measure='controllability', pooling_sizes=None):
    node_importance = cal_node_importance(graph, measure=measure)
    for i in graph.nodes:
        graph.nodes[i]['importance'] = node_importance[i]
    ranking_nodes = sorted(nx.degree(graph), key=lambda x: x[1], reverse=True)
    ranking_nodes_id = [n[0] for n in ranking_nodes]
    rank_tensor = []
    for i in ranking_nodes_id:
        rank_tensor.append(graph.nodes[i]['importance'])
    rank_tensor = np.array(rank_tensor).T
    pooling_tensor = pyramid_pooling(rank_tensor, pooling_sizes=pooling_sizes, pooling_way='mean')
    return pooling_tensor


def cal_node_importance(graph: nx.Graph, max_hop=3, measure='controllability'):
    importance = []
    degrees = np.array(nx.degree(graph))
    for k in range(1, max_hop + 1):
        temp_importance = []
        temp_importance1 = []
        subgraphs = extract_k_hop_subgraphs(graph, k)
        if measure != 'both':
            for node, g in zip(graph.nodes, subgraphs):
                temp_importance.append(node_importance_in_subgraph(node, g, measure=measure))
            importance.append(temp_importance)
        else:
            for node, g in zip(graph.nodes, subgraphs):
                temp_importance.append(node_importance_in_subgraph(node, g, measure='controllability'))
            importance.append(temp_importance)
            for node, g in zip(graph.nodes, subgraphs):
                temp_importance1.append(node_importance_in_subgraph(node, g, measure='connectivity'))
            importance.append(temp_importance1)
    node_feats = np.concatenate((degrees[:, 1:2], np.array(importance).T), axis=1)
    scaler = MinMaxScaler()
    node_feats = scaler.fit_transform(node_feats)
    return node_feats


def extract_k_hop_subgraphs(graph: nx.Graph, k=2):
    subgraphs = [nx.ego_graph(graph, i, radius=k) for i in graph.nodes]
    return subgraphs


def node_importance_in_subgraph(node, subgraph: networkx.Graph, measure='controllability'):
    attacked_graph = subgraph.copy()
    attacked_graph.remove_node(node)
    if measure == 'controllability':
        if not attacked_graph.nodes:
            return 0
        c_original = cal_controllability(subgraph)
        c_attacked = cal_controllability(attacked_graph)
        return abs(c_original - c_attacked)
    if measure == 'connectivity':
        if not attacked_graph.nodes:
            return 0
        c_original = cal_connectivity(subgraph)
        c_attacked = cal_connectivity(attacked_graph)
        return abs(c_original - c_attacked)
