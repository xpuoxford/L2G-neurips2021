
import numpy as np
import math

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import euclidean_distances
import scipy.sparse as sparse
from sklearn import metrics
import scipy.stats


#%%

def halfvec_to_topo(w, threshold, device):
    """
    from half vectorisation to matrix in batch way.
    """

    batch_size, l = w.size()
    m = int((1 / 2) * (1 + math.sqrt(1 + 8 * l)))

    # extract binary edge {0, 1}:
    bw = (w.clone().detach() >= threshold).float().to(device)
    E = torch.zeros((batch_size, m, m), dtype = w.dtype).to(device)

    for i in range(batch_size):
        E[i, :, :][np.triu_indices(m, 1)] = bw[i].clone().detach()
        E[i, :, :] = E[i, :, :].T + E[i, :, :]

    return E

#%%

def torch_sqaureform_to_matrix(w, device):
    """
    from half vectorisation to matrix in batch way.
    """

    batch_size, l = w.size()
    m = int((1 / 2) * (1 + math.sqrt(1 + 8 * l)))

    E = torch.zeros((batch_size, m, m), dtype = w.dtype).to(device)

    for i in range(batch_size):
        E[i, :, :][np.triu_indices(m, 1)] = w[i].clone().detach()
        E[i, :, :] = E[i, :, :].T + E[i, :, :]

    return E

#%%

def torch_squareform_to_vector(A, device):
    batch_size, m, _ = A.size()
    l = int(m * (m - 1) / 2)

    w = torch.zeros((batch_size, l), dtype = A.dtype).to(device)

    for i in range(batch_size):
        w[i, :] = A[i,:,:][np.triu_indices(m, 1)].clone().detach()

    return w

#%%

def soft_threshold(w, eta):
    '''
    softthreshold function in a batch way.
    '''
    return (torch.abs(w) >= eta) * torch.sign(w) * (torch.abs(w) - eta)


#%%

def check_tensor(x, device):
    if isinstance(x, np.ndarray) or type(x) in [int, float]:
        x = torch.Tensor(x)
    if isinstance(x, torch.Tensor):
        return x.to(device=device)
    return x

#%%

def coo_to_sparseTensor(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()


#%%

def get_degree_operator(m):
    ncols =int(m*(m - 1)/2)

    I = np.zeros(ncols)
    J = np.zeros(ncols)

    k = 0
    for i in np.arange(1, m):
        I[k:(k + m - i)] = np.arange(i, m)
        k = k + (m - i)

    k = 0
    for i in np.arange(1, m):
        J[k: (k + m - i)] = i - 1
        k = k + m - i

    Row = np.tile(np.arange(0, ncols), 2)
    Col = np.append(I, J)
    Data = np.ones(Col.size)
    St = sparse.coo_matrix((Data, (Row, Col)), shape=(ncols, m))
    return St.T

#%%

def get_distance_halfvector(y):
    n, _ = y.shape # m nodes, n observations
    z = (1 / n) * euclidean_distances(y.T, squared=True)
    # z.shape = m, m
    return squareform(z, checks=False)

#%%

def acc_loss(w_list, w, dn=0.9):
    num_unrolls, _ = w_list.size()

    if dn is None:
        # no accumulation, only the last unroll result
        loss = torch.sum((w_list[num_unrolls - 1, :] - w) ** 2) / torch.sum(w ** 2)

    elif dn == 1:
        # all layer matters
        loss = sum(torch.sum((w_list - w) ** 2, dim=1) / torch.sum(w ** 2))

    else:
        # cumulative loss
        factor = torch.tensor([dn ** i for i in range(num_unrolls, 0, -1)]).to(device)
        loss = sum(factor * torch.sum((w_list - w) ** 2, dim=1) / torch.sum(w ** 2))

    return loss

def gmse_loss(w_pred, w):
    loss = torch.sum((w_pred - w) ** 2) / torch.sum(w ** 2)
    return loss

def gmse_loss_batch_mean(w_pred, w):
    loss = torch.sum((w_pred - w) ** 2, dim = -1) / torch.sum(w ** 2, dim = -1)
    return loss.mean()

def gmse_loss_batch(w_pred, w):
    loss = torch.sum((w_pred - w) ** 2, dim = -1) / torch.sum(w ** 2, dim = -1)
    return loss

def layerwise_gmse_loss(w_list, w):
    num_unrolls, _ = w_list.size()
    loss = torch.sum((w_list - w) ** 2, dim=1) / torch.sum(w ** 2)
    return loss

#%%


def get_auc(y, scores):
    y = np.array(y).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    roc_auc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(y, scores)
    return roc_auc, aupr


def report_metrics(G_true, G, beta=1):
    d = G.shape[-1]

    G_binary = np.where(G!=0, 1, 0)
    G_true_binary = np.where(G_true!=0, 1, 0)

    # extract the upper diagonal matrix
    indices_triu = np.triu_indices(d, 1)
    edges_true = G_true_binary[indices_triu]
    edges_pred = G_binary[indices_triu]

    # Getting AUROC value
    edges_pred_auc = G[indices_triu]
    auc, aupr = get_auc(edges_true, np.absolute(edges_pred_auc))

    TP = np.sum(edges_true * edges_pred)
    mismatches = np.logical_xor(edges_true, edges_pred)
    FP = np.sum(mismatches * edges_pred)
    P = np.sum(edges_pred)
    T = np.sum(edges_true)
    F = len(edges_true) - T
    SHD = np.sum(mismatches)
    FDR = FP/P
    TPR = TP/T
    FPR = FP/F
    FN = np.sum(mismatches * edges_true)
    num = (1+beta**2)*TP
    den = ((1+beta**2)*TP + beta**2 * FN + FP)
    F_beta = num/den
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    result = {
        'FDR': FDR,
        'TPR': TPR,
        'FPR': FPR,
        'SHD': SHD,
        'T': T,
        'P': P,
        'precision': precision,
        'recall': recall,
        'F_beta': F_beta,
        'aupr': aupr,
        'auc': auc
    }

    return result



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n- 1)
    return m, h, m - h, m + h


def binary_metrics_batch(adj_true_batch, w_pred_batch, device1):
    G_pred_batch = torch_sqaureform_to_matrix(w_pred_batch, device=device).detach().cpu()
    G_true_batch = torch_sqaureform_to_matrix(adj_true_batch, device=device).detach().cpu()

    batch_size = G_pred_batch.size()[0]

    AUC = [report_metrics(G_true_batch[i, :, :], G_pred_batch[i, :, :], beta=1)['auc'] for i in range(batch_size)]
    auc_mean, auc_ci, _, _ = mean_confidence_interval(np.array(AUC), confidence=0.95)

    APS = [report_metrics(G_true_batch[i, :, :], G_pred_batch[i, :, :], beta=1)['aupr'] for i in range(batch_size)]
    aps_mean, aps_ci, _, _ = mean_confidence_interval(np.array(APS), confidence=0.95)

    result = {
        'auc_mean': auc_mean,
        'auc_ci': auc_ci,
        'aps_mean': aps_mean,
        'aps_ci': aps_ci
    }

    return result

#%%