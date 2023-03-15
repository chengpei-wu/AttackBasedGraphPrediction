import networkx
import networkx as nx
import numpy as np

from utils.pyramid_pooling_utils import pyramid_pooling


def attack_sim(graph: networkx.Graph, measure='controllability', fixed='pooling', pooling_size=None, sample_size=None):
    if measure not in ['controllability', 'connectivity']:
        print(f'{measure} not yet implemented')
        return

    if measure == 'controllability':
        c_origin = cal_controllability(graph)
        curve = [c_origin]
        while graph.number_of_nodes() > 1:
            attack_id = get_target_id(graph)
            graph.remove_node(attack_id)
            curve.append(cal_controllability(graph))
    if measure == 'connectivity':
        c_origin = cal_connectivity(graph)
        curve = [c_origin]
        while graph.number_of_nodes() > 1:
            attack_id = get_target_id(graph)
            graph.remove_node(attack_id)
            curve.append(cal_connectivity(graph))
    if fixed == 'sample':
        return sample_curve(np.array(curve), sample_size)
    else:
        return pyramid_pooling([np.array(curve)], pooling_sizes=pooling_size, pooling_way='mean')


def cal_controllability(graph: networkx.Graph):
    num_nodes = graph.number_of_nodes()
    A = nx.to_numpy_matrix(graph)
    rank_A = np.linalg.matrix_rank(A)
    return max([1, num_nodes - rank_A]) / num_nodes


def cal_connectivity(graph: networkx.Graph):
    num_nodes = graph.number_of_nodes()
    # 计算连通子图
    largest_connected_subgraph = max(nx.connected_components(graph), key=len)
    return len(largest_connected_subgraph) / num_nodes


def get_target_id(graph: networkx.Graph):
    betweenness_centrality = nx.betweenness_centrality(graph)
    clustering = nx.clustering(graph)
    degrees = dict(graph.degree())
    max_degree = max(degrees.values())
    max_degree_nodes = [node for node, degree in degrees.items() if degree == max_degree]

    def custom_sort(node):
        betweenness = betweenness_centrality[node]
        cluster_coefficient = clustering[node]
        return -betweenness, -cluster_coefficient

    if len(max_degree_nodes) == 1:
        attack_id = max_degree_nodes[0]
    else:
        sorted_nodes = sorted(max_degree_nodes, key=custom_sort)
        attack_id = sorted_nodes[0]
    return attack_id


def sample_curve(curve, max_length=30):
    _, positions = np.histogram(range(len(curve)), max_length)
    positions = np.round(positions).astype(np.int32)
    sampled_curve = curve[positions]
    return sampled_curve
