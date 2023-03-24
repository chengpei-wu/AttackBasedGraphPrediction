import networkx
import networkx as nx
import numpy as np

from utils.pyramid_pooling_utils import pyramid_pooling


def get_graph_tensor_from_graph_attack(graph: networkx.Graph, measures=None, fixed='pooling',
                                       pooling_sizes=None, sample_size=4):
    if measures is None:
        measures = ['controllability', 'connectivity']
    curves = []
    _, attack_sequence = get_target_id(graph)
    for measure in measures:
        graph_ = graph.copy()
        if measure == 'controllability':
            c_origin = cal_controllability(graph_)
            curve = [c_origin]
            for attack_id in attack_sequence[:-1]:
                graph_.remove_node(attack_id)
                curve.append(cal_controllability(graph_))
            curves.append(curve)

        if measure == 'connectivity':
            c_origin = cal_connectivity(graph_)
            curve = [c_origin]
            for attack_id in attack_sequence[:-1]:
                graph_.remove_node(attack_id)
                curve.append(cal_connectivity(graph_))
            curves.append(curve)

        if measure == 'average_degree':
            c_origin = cal_average_degree(graph_)
            curve = [c_origin]
            for attack_id in attack_sequence[:-1]:
                graph_.remove_node(attack_id)
                curve.append(cal_average_degree(graph_))
            curves.append(curve)

        if measure == 'betweenness':
            c_origin = cal_betweenness(graph_)
            curve = [c_origin]
            for attack_id in attack_sequence[:-1]:
                graph_.remove_node(attack_id)
                curve.append(cal_betweenness(graph_))
            curves.append(curve)

        if measure == 'clustering':
            c_origin = cal_clustering_coefficient(graph_)
            curve = [c_origin]
            for attack_id in attack_sequence[:-1]:
                graph_.remove_node(attack_id)
                curve.append(cal_clustering_coefficient(graph_))
            curves.append(curve)

        if measure == 'communication_ability':
            c_origin = cal_communication_ability(graph_)
            curve = [c_origin]
            for attack_id in attack_sequence[:-1]:
                graph_.remove_node(attack_id)
                curve.append(cal_communication_ability(graph_))
            curves.append(curve)

        if measure == 'spectral_radius':
            c_origin = cal_adj_spectral_radius(graph_)
            curve = [c_origin]
            for attack_id in attack_sequence[:-1]:
                graph_.remove_node(attack_id)
                curve.append(cal_adj_spectral_radius(graph_))
            curves.append(curve)

        if measure == 'spectral_gap':
            c_origin = cal_adj_spectral_gap(graph_)
            curve = [c_origin]
            for attack_id in attack_sequence[:-1]:
                graph_.remove_node(attack_id)
                curve.append(cal_adj_spectral_gap(graph_))
            curves.append(curve)

        if measure == 'natural_connectivity':
            c_origin = cal_natural_connectivity(graph_)
            curve = [c_origin]
            for attack_id in attack_sequence[:-1]:
                graph_.remove_node(attack_id)
                curve.append(cal_natural_connectivity(graph_))
            curves.append(curve)

        if measure == 'algebraic_connectivity':
            c_origin = cal_algebraic_connectivity(graph_)
            curve = [c_origin]
            for attack_id in attack_sequence[:-1]:
                graph_.remove_node(attack_id)
                curve.append(cal_algebraic_connectivity(graph_))
            curves.append(curve)

    if fixed == 'sample':
        return sample_curve(np.array(curves), max_length=sample_size)
    else:
        pp_curve = pyramid_pooling(np.array(curves), pooling_sizes=pooling_sizes, pooling_way='mean')
        return pp_curve


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


def cal_average_degree(graph: networkx.Graph):
    degrees = dict(nx.degree(graph)).values()
    return sum(degrees) / graph.number_of_nodes()


def cal_betweenness(graph: networkx.Graph):
    bets = dict(nx.betweenness_centrality(graph)).values()
    return sum(bets) / graph.number_of_nodes()


def cal_clustering_coefficient(graph: networkx.Graph):
    cc = nx.average_clustering(graph)
    return cc


def cal_communication_ability(graph: networkx.Graph):
    s = [(len(c) ** 2) for c in nx.connected_components(graph)]
    return sum(s) / (graph.number_of_nodes() ** 2)


def cal_adj_spectral_radius(graph: networkx.Graph):
    adj = nx.to_numpy_matrix(graph)
    a, b = np.linalg.eig(adj)
    return np.max(np.abs(a))


def cal_adj_spectral_gap(graph: networkx.Graph):
    adj = nx.to_numpy_matrix(graph)
    a, b = np.linalg.eig(adj)
    a = np.sort(np.abs(a))
    return a[-1] - a[-2]


def cal_natural_connectivity(graph: networkx.Graph):
    adj = nx.to_numpy_matrix(graph)
    a, b = np.linalg.eig(adj)
    return np.mean(np.abs(a))


def cal_algebraic_connectivity(graph: networkx.Graph):
    if graph.number_of_nodes() <= 2:
        return 0
    return nx.linalg.algebraic_connectivity(graph)


def cal_effective_resistance(graph: networkx.Graph):
    adj = nx.to_numpy_matrix(graph)
    degrees = [d[1] for d in nx.degree(graph)]
    deg = np.diag(degrees)
    lapla = deg - adj
    a, b = np.linalg.eig(lapla)
    return np.sum(a) / graph.number_of_nodes()


def cal_number_of_spanning_trees(graph: networkx.Graph):
    adj = nx.to_numpy_matrix(graph)
    degrees = [d[1] for d in nx.degree(graph)]
    deg = np.diag(degrees)
    lapla = deg - adj
    a, b = np.linalg.eig(lapla)
    return np.cumprod(a)[-1] / graph.number_of_nodes()


def get_target_id(graph: networkx.Graph):
    betweenness_centrality = nx.betweenness_centrality(graph)
    clustering = nx.clustering(graph)
    degrees = np.array([n[1] for n in nx.degree(graph)])
    all_nodes = graph.nodes

    def custom_sort(node):
        degree = degrees[node]
        betweenness = betweenness_centrality[node]
        cluster_coefficient = clustering[node]
        return -degree, -betweenness, -cluster_coefficient

    sorted_nodes = sorted(all_nodes, key=custom_sort)
    attack_id = sorted_nodes[0]
    return attack_id, sorted_nodes


def sample_curve(curves, max_length=30):
    sampled_curves = []
    for curve in curves:
        _, positions = np.histogram(range(len(curve)), max_length)
        positions = np.round(positions).astype(np.int32)
        sampled_curve = curve[positions[:-1]]
        sampled_curves = np.concatenate((sampled_curves, sampled_curve))
    return sampled_curves
