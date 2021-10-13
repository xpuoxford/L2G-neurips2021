
import pickle
from src.utils_data import *

#%%

graph_type = 'BA'
edge_type = 'lognormal'
graph_size = 20

graph_hyper = 3

data = generate_BA_parallel(num_samples=8064,
                            num_signals=3000,
                            num_nodes=graph_size,
                            graph_hyper=graph_hyper,
                            weighted=edge_type,
                            weight_scale=True)

with open('data/dataset_{}_{}nodes.pickle'.format(graph_type, graph_size), 'wb') as handle:
    pickle.dump(data, handle, protocol=4)

#%%