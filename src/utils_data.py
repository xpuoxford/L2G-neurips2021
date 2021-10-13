
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split

import networkx as nx
import scipy
import pickle
import multiprocess
from functools import partial

from src.utils import *

#%%
def data_loading(dir_dataset, batch_size = None, train_prop=0.8):

    with open(dir_dataset, 'rb') as handle:
        dataset = pickle.load(handle)

    print('loading data at ', dir_dataset)

    num_samples = len(dataset['z'])
    w = [squareform(dataset['W'][i].A) for i in range(num_samples)]

    test_size = 64
    num_samples -= 64
    train_size = int(train_prop * num_samples)
    val_size = int(num_samples - train_size)

    data = TensorDataset(torch.Tensor(dataset['z']), torch.Tensor(w))
    train_data, val_data, test_data = random_split(data, [train_size, val_size, test_size])

    if batch_size is not None:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=test_size, shuffle=True)
        print('successfully loading: train {}, val {}, test {}, batch {}'.format(train_size, val_size,
                                                                                 test_size, batch_size))
        return train_loader, val_loader, test_loader

    else:
        print('successfully loading: train size {}, val size {}, test size {}'.format(train_size, val_size, test_size))
        return train_data, val_data, test_data

#%%

def test_data_loading(dir_dataset):

    with open(dir_dataset, 'rb') as handle:
        dataset = pickle.load(handle)

    print('loading data at ', dir_dataset)

    num_samples = len(dataset['z'])
    w = [squareform(dataset['W'][i].A) for i in range(num_samples)]

    test_size = 64
    test_data = TensorDataset(torch.Tensor(dataset['z']), torch.Tensor(w))
    test_loader = DataLoader(test_data, batch_size=test_size, shuffle=False)

    return test_loader

#%%

def _generate_BA_to_parallel(i, num_nodes, num_signals, graph_hyper, weighted, weight_scale = False):

    G = nx.barabasi_albert_graph(num_nodes, graph_hyper)

    W_GT = nx.adjacency_matrix(G).A

    if weighted == 'uniform':
        weights = np.random.uniform(0, 2, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'gaussian':
        weights = np.random.normal(1, 0.05, (num_nodes, num_nodes))
        weights = np.abs(weights)
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'lognormal':
        weights = np.random.lognormal(0, 0.1, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights


    if weight_scale:
        W_GT = W_GT * num_nodes / np.sum(W_GT)

    L_GT = np.diag(W_GT @ np.ones(num_nodes)) - W_GT

    W_GT = scipy.sparse.csr_matrix(W_GT)

    cov = np.linalg.inv(L_GT + (1e-04) * np.eye(num_nodes))
    z = get_distance_halfvector(np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals))

    return z, W_GT

def generate_BA_parallel(num_samples, num_nodes, num_signals, graph_hyper, weighted, weight_scale):
    n_cpu = multiprocess.cpu_count() - 2
    pool = multiprocess.Pool(n_cpu)

    z_multi, W_multi = zip(*pool.map(partial(_generate_BA_to_parallel,
                                             num_nodes = num_nodes,
                                             num_signals = num_signals,
                                             graph_hyper = graph_hyper,
                                             weighted = weighted,
                                             weight_scale = weight_scale),
                                     range(num_samples)))

    result = {
        'z': z_multi,
        'W': W_multi
    }

    return result

#%%



def _generate_WS_to_parallel(i, num_nodes, num_signals, graph_hyper, weighted, weight_scale = False):

    G = nx.watts_strogatz_graph(num_nodes, k = graph_hyper['k'], p = graph_hyper['p'])

    W_GT = nx.adjacency_matrix(G).A

    if weighted == 'uniform':
        weights = np.random.uniform(0, 2, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'gaussian':
        weights = np.random.normal(1, 0.05, (num_nodes, num_nodes))
        weights = np.abs(weights)
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'lognormal':
        weights = np.random.lognormal(0, 0.1, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights


    if weight_scale:
        W_GT = W_GT * num_nodes / np.sum(W_GT)

    L_GT = np.diag(W_GT @ np.ones(num_nodes)) - W_GT

    W_GT = scipy.sparse.csr_matrix(W_GT)

    cov = np.linalg.inv(L_GT + (1e-04) * np.eye(num_nodes))
    #signal = np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals)
    #z = get_distance_halfvector(signal)

    # signal = np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals)
    z = get_distance_halfvector(np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals))

    return z, W_GT

def generate_WS_parallel(num_samples, num_nodes, num_signals, graph_hyper, weighted, weight_scale):
    n_cpu = multiprocess.cpu_count() - 2
    pool = multiprocess.Pool(n_cpu)

    z_multi, W_multi = zip(*pool.map(partial(_generate_WS_to_parallel,
                                             num_nodes = num_nodes,
                                             num_signals = num_signals,
                                             graph_hyper = graph_hyper,
                                             weighted = weighted,
                                             weight_scale = weight_scale),
                                     range(num_samples)))

    result = {
        'z': z_multi,
        'W': W_multi
    }

    return result

#%%


def _generate_ER_to_parallel(i, num_nodes, num_signals, graph_hyper, weighted, weight_scale = False):

    G = nx.erdos_renyi_graph(num_nodes, graph_hyper)

    W_GT = nx.adjacency_matrix(G).A

    if weighted == 'uniform':
        weights = np.random.uniform(0, 2, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'gaussian':
        weights = np.random.normal(1, 1e-02, (num_nodes, num_nodes))
        weights = np.abs(weights)
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'lognormal':
        weights = np.random.lognormal(0, 0.1, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights


    if weight_scale:
        W_GT = W_GT * num_nodes / np.sum(W_GT)

    L_GT = np.diag(W_GT @ np.ones(num_nodes)) - W_GT

    W_GT = scipy.sparse.csr_matrix(W_GT)

    cov = np.linalg.inv(L_GT + (1e-04) * np.eye(num_nodes))
    z = get_distance_halfvector(np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals))

    return z, W_GT

def generate_ER_parallel(num_samples, num_nodes, num_signals, graph_hyper, weighted, weight_scale):
    n_cpu = multiprocess.cpu_count() - 2
    pool = multiprocess.Pool(n_cpu)

    z_multi, W_multi = zip(*pool.map(partial(_generate_ER_to_parallel,
                                             num_nodes = num_nodes,
                                             num_signals = num_signals,
                                             graph_hyper = graph_hyper,
                                             weighted = weighted,
                                             weight_scale = weight_scale),
                                     range(num_samples)))

    result = {
        'z': z_multi,
        'W': W_multi
    }

    return result

#%%


def _generate_SBM100noise_to_parallel(i, num_nodes, num_signals, graph_hyper, weighted, weight_scale = False):

    size = [4, 2, 2, 13, 13, 15, 17, 3, 12, 10, 9]

    p = graph_hyper
    probs = [[0.95, p, p, p, p, p, p, p, p, p, p],
             [p, 1, p, p, p, p, p, p, p, p, p],
             [p, p, 1, p, p, p, p, p, p, p, p],
             [p, p, p, 0.6, p, p, p, p, p, p, p],
             [p, p, p, p, 0.6, p, p, p, p, p, p],
             [p, p, p, p, p, 0.5, p, p, p, p, p],
             [p, p, p, p, p, p, 0.5, p, p, p, p],
             [p, p, p, p, p, p, p, 0.95, p, p, p],
             [p, p, p, p, p, p, p, p, 0.65, p, p],
             [p, p, p, p, p, p, p, p, p, 0.65, p],
             [p, p, p, p, p, p, p, p, p, p, 0.65]]

    G = nx.stochastic_block_model(size, probs)

    W_GT = nx.adjacency_matrix(G).A

    if weighted == 'uniform':
        weights = np.random.uniform(0, 2, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'gaussian':
        weights = np.random.normal(1, 0.05, (num_nodes, num_nodes))
        weights = np.abs(weights)
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'lognormal':
        weights = np.random.lognormal(0, 0.1, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights


    if weight_scale:
        W_GT = W_GT * num_nodes / np.sum(W_GT)

    L_GT = np.diag(W_GT @ np.ones(num_nodes)) - W_GT

    W_GT = scipy.sparse.csr_matrix(W_GT)

    cov = np.linalg.inv(L_GT + (1e-06) * np.eye(num_nodes))
    z = get_distance_halfvector(np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals))

    return z, W_GT

def generate_SBM100noise_parallel(num_samples, num_nodes, num_signals, graph_hyper, weighted, weight_scale):
    n_cpu = multiprocess.cpu_count() - 2
    pool = multiprocess.Pool(n_cpu)

    z_multi, W_multi = zip(*pool.map(partial(_generate_SBM100noise_to_parallel,
                                             num_nodes = num_nodes,
                                             num_signals = num_signals,
                                             graph_hyper = graph_hyper,
                                             weighted = weighted,
                                             weight_scale = weight_scale),
                                     range(num_samples)))

    result = {
        'z': z_multi,
        'W': W_multi
    }

    return result

#%%
