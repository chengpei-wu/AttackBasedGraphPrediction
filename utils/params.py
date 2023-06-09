pooling_sizes = {
    'MUTAG': [1, 2, 4, 8],
    'PTC_MR': [1, 2, 4, 8, 16],
    'NCI1': [1, 2, 4, 8, 16, 32],
    'PROTEINS': [1, 2, 4, 8, 16, 40],
    'DD': [1, 2, 4, 8, 16, 32, 64, 128, 256, 400],
    'COLLAB': [1, 2, 4, 8, 16, 32, 64, 128],
    'IMDB-BINARY': [1, 2, 4, 8, 16, 32],
    'IMDB-MULTI': [1, 2, 4, 8, 16],
    'REDDIT-BINARY': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    'REDDIT-MULTI-5K': [1, 2, 4, 8, 16, 32, 128, 256, 512, 780],
}

attack_measures = [
    'controllability',
    'connectivity',
    'average_degree',
    'betweenness',
    'clustering',
    'communication_ability',
    'spectral_radius',
    # 'spectral_gap',
    'natural_connectivity',
    'algebraic_connectivity',
]
