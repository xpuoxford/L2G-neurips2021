
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from src.utils import *

#%%

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        if bias is True:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features))
            nn.init.ones_(self.bias)
        else:
            self.bias = None

    def forward(self, node_feature, adj):
        h = torch.matmul(node_feature, self.weight)
        output = torch.bmm(adj, h)
        if self.bias is not None:
            return output + self.bias
        return output

#%%

class GraphEnc(nn.Module):
    def __init__(self, n_inNodeFeat, n_hid, n_outGraphFeat):
        super(GraphEnc, self).__init__()
        self.conv1 = GraphConvLayer(in_features=n_inNodeFeat, out_features=n_hid, bias=True)
        self.conv2 = GraphConvLayer(in_features=n_hid, out_features=int(n_hid * 2), bias=True)
        self.fc1 = nn.Linear(int(n_hid * 2), n_hid, bias=True)
        self.fc2 = nn.Linear(n_hid, n_outGraphFeat, bias=True)

    def forward(self, adj):
        degree = torch.sum(adj, dim=-1).unsqueeze_(-1).to(device)

        # 2-layer GCN
        h = F.relu(self.conv1(degree, adj))
        h = F.relu(self.conv2(h, adj))

        # readout mean of all node embeddings
        hg = torch.sum(h, dim=1)
        hg = F.relu(self.fc1(hg))
        hg = self.fc2(hg)

        return hg


#%%

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('tanh'))
        torch.nn.init.zeros_(m.bias)

#%%

class TopoDiffVAE(nn.Module):
    def __init__(self, graph_size, n_hid, n_latent, n_nodeFeat, n_graphFeat):
        super(TopoDiffVAE, self).__init__()

        self.enc = GraphEnc(n_nodeFeat, n_hid, n_graphFeat)

        self.f_mean = nn.Sequential(
            nn.Linear(n_graphFeat, n_latent)
        )
        self.f_mean.apply(init_weights)

        self.f_var = nn.Sequential(
            nn.Linear(n_graphFeat, n_latent)
        )
        self.f_var.apply(init_weights)

        self.dec = nn.Sequential(
            nn.Linear(int(graph_size * (graph_size - 1) / 2) + n_latent, int(graph_size * (graph_size - 1)*2/3)),
            nn.Tanh(),
            nn.Linear(int(graph_size * (graph_size - 1)*2/3), int(graph_size * (graph_size - 1)/2)),
        )
        self.dec.apply(init_weights)

        self.n_latent = n_latent

    def resample(self, z_vecs, f_mean, f_var):

        z_mean = f_mean(z_vecs)
        z_log_var = -torch.abs(f_var(z_vecs))

        kl_loss = -0.5 * torch.mean(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var))

        epsilon = torch.randn_like(z_mean).to(device)  # N(0,1) in [batch_size, n_latent]
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon  # latent variable in [batch_size, n_latent]

        return z_vecs, kl_loss

    def latent_noise(self, x_emb, x_batch):

        batch_size = x_emb.size(0)

        diff_noise = torch.randn(batch_size, self.n_latent).to(device)
        latent = torch.cat([x_batch, diff_noise], dim=-1)

        return latent

    def latent_diff(self, x_emb, y_emb, x_batch):
        diff = y_emb - x_emb

        topo_diff_latent, kl = self.resample(diff, self.f_mean, self.f_var)
        latent = torch.cat([x_batch, topo_diff_latent], dim=-1)

        return latent, kl

    def forward(self, x_batch, y_batch, threshold=1e-04, kl_hyper=1):

        # binarification
        x_topo = halfvec_to_topo(x_batch, threshold, device=device)
        y_topo = halfvec_to_topo(y_batch, threshold, device=device)

        # encoding:
        x_emb = self.enc(x_topo)
        y_emb = self.enc(y_topo)

        # latent:
        latent, kl = self.latent_diff(x_emb, y_emb, x_batch)

        # decoding:
        y_pred = self.dec(latent)

        # loss:
        recons_loss = torch.sum((y_pred - y_batch) ** 2, dim=-1).mean()
        loss = recons_loss + kl_hyper * kl

        return y_pred, loss, kl, latent

    def refine(self, x_batch, threshold=1e-04):
        """
        validation
        """
        x_topo = halfvec_to_topo(x_batch, threshold, device=device)

        x_emb = self.enc(x_topo)
        latent = self.latent_noise(x_emb, x_batch)
        y_pred = self.dec(latent)

        return y_pred, latent

#%%

class learn2graph(nn.Module):
    """
    main module for L2G.
    """
    def __init__(self, num_unroll, graph_size, n_hid, n_latent, n_nodeFeat, n_graphFeat):
        super(learn2graph, self).__init__()

        self.layers = num_unroll
        self.gn = nn.Parameter(torch.ones(num_unroll, 1)/100, requires_grad=True)

        self.beta = nn.Parameter(torch.FloatTensor(num_unroll, 1), requires_grad=True)
        nn.init.ones_(self.beta)

        self.alpha = nn.Parameter(torch.FloatTensor(num_unroll, 1), requires_grad=True)
        nn.init.ones_(self.alpha)

        self.vae = TopoDiffVAE(graph_size, n_hid, n_latent, n_nodeFeat, n_graphFeat)

    def prox_log_barrier(self, y, gn, alpha):
        up = y ** 2 + 4 * gn * alpha
        up = torch.clamp(up, min=1e-08)
        return (y - torch.sqrt(up)) / 2

    def initialisation(self, m, l, batch_size):
        w = torch.zeros((batch_size, l)).float().to(device)
        v = torch.zeros((batch_size, m)).float().to(device)
        return w, v

    def forward(self,z, w_gt_batch, threshold=1e-04, kl_hyper=1):
        batch_size, l = z.size()

        z = check_tensor(z, device)
        m = int((1 / 2) * (1 + math.sqrt(1 + 8 * l)))
        D = coo_to_sparseTensor(get_degree_operator(m)).to(device)

        # initialise:
        w, v = self.initialisation(m, l, batch_size)
        w_list = torch.empty(size=(batch_size, self.layers, l)).to(device)

        for i in range(int(self.layers-1)):

            y1 = w - self.gn[i] * (2 * self.beta[i] * w + 2 * z + torch.matmul(v, D))
            y2 = v + self.gn[i] * torch.matmul(w, D.T)

            p1 = F.leaky_relu(y1)
            p2 = self.prox_log_barrier(y2, self.gn[i], self.alpha[i])

            q1 = p1 - self.gn[i] * (2 * self.beta[i] * p1 + 2 * z + torch.matmul(p2, D))
            q2 = p2 + self.gn[i] * torch.matmul(p1, D.T)

            w = w - y1 + q1
            v = v - y2 + q2

            w_list[:, i, :] = w

        i += 1
        y1 = w - self.gn[i] * (2 * self.beta[i] * w + 2 * z + torch.matmul(v, D))
        y2 = v + self.gn[i] * torch.matmul(w, D.T)

        p1, vae_loss, vae_kl, vae_latent = self.vae.forward(y1, w_gt_batch, threshold, kl_hyper)
        p2 = self.prox_log_barrier(y2, self.gn[i], self.alpha[i])

        q1 = p1 - self.gn[i] * (2 * self.beta[i] * p1 + 2 * z + torch.matmul(p2, D))
        q2 = p2 + self.gn[i] * torch.matmul(p1, D.T)

        w = w - y1 + q1
        v = v - y2 + q2

        w_list[:, i, :] = w

        return w_list, vae_loss, vae_kl, vae_latent

    def validation(self, z, threshold=1e-04):
        batch_size, l = z.size()

        z = check_tensor(z, device)
        m = int((1 / 2) * (1 + math.sqrt(1 + 8 * l)))
        D = coo_to_sparseTensor(get_degree_operator(m)).to(device)

        # initialise:
        w, v = self.initialisation(m, l, batch_size)
        w_list = torch.empty(size=(batch_size, self.layers, l)).to(device)

        for i in range(int(self.layers - 1)):
            y1 = w - self.gn[i] * (2 * self.beta[i] * w + 2 * z + torch.matmul(v, D))
            y2 = v + self.gn[i] * torch.matmul(w, D.T)

            p1 = F.leaky_relu(y1)
            p2 = self.prox_log_barrier(y2, self.gn[i], self.alpha[i])

            q1 = p1 - self.gn[i] * (2 * self.beta[i] * p1 + 2 * z + torch.matmul(p2, D))
            q2 = p2 + self.gn[i] * torch.matmul(p1, D.T)

            w = w - y1 + q1
            v = v - y2 + q2

            w_list[:, i, :] = w

        i += 1
        y1 = w - self.gn[i] * (2 * self.beta[i] * w + 2 * z + torch.matmul(v, D))
        y2 = v + self.gn[i] * torch.matmul(w, D.T)

        p1, _ = self.vae.refine(y1, threshold)
        p2 = self.prox_log_barrier(y2, self.gn[i], self.alpha[i])

        q1 = p1 - self.gn[i] * (2 * self.beta[i] * p1 + 2 * z + torch.matmul(p2, D))
        q2 = p2 + self.gn[i] * torch.matmul(p1, D.T)

        w = w - y1 + q1
        v = v - y2 + q2

        w_list[:, i, :] = w

        return w_list


#%%

class unrolling(nn.Module):
    """
    ablation model: unrolling

    """

    def __init__(self, num_unroll):
        super(unrolling, self).__init__()

        self.layers = num_unroll  # int

        self.gn = nn.Parameter(torch.ones(num_unroll, 1)/100, requires_grad=True)

        self.beta = nn.Parameter(torch.FloatTensor(num_unroll, 1), requires_grad=True)
        nn.init.ones_(self.beta)

        self.alpha = nn.Parameter(torch.FloatTensor(num_unroll, 1), requires_grad=True)
        nn.init.ones_(self.alpha)

    def prox_log_barrier(self, y, gn, alpha):
        up = y ** 2 + 4 * gn * alpha
        up = torch.clamp(up, min = 1e-08)
        return (y - torch.sqrt(up)) / 2

    def initialisation(self, m, l, batch_size):
        w = torch.zeros((batch_size, l)).float().to(device)
        v = torch.zeros((batch_size, m)).float().to(device)
        return w, v

    def forward(self, z):
        batch_size, l = z.size()

        z = check_tensor(z, device)
        m = int((1 / 2) * (1 + math.sqrt(1 + 8 * l)))
        D = coo_to_sparseTensor(get_degree_operator(m)).to(device)

        # initialise:
        w, v = self.initialisation(m, l, batch_size)
        zero_vec = torch.zeros((batch_size, l)).to(device)
        w_list = torch.empty(size=(batch_size, self.layers, l)).to(device)

        for i in range(self.layers):

            y1 = w - self.gn[i] * (2 * self.beta[i] * w + 2 * z + torch.matmul(v, D))
            y2 = v + self.gn[i] * torch.matmul(w, D.T)

            p1 = torch.max(zero_vec, y1)
            p2 = self.prox_log_barrier(y2, self.gn[i], self.alpha[i])

            q1 = p1 - self.gn[i] * (2 * self.beta[i] * p1 + 2 * z + torch.matmul(p2, D))
            q2 = p2 + self.gn[i] * torch.matmul(p1, D.T)

            w = w - y1 + q1
            v = v - y2 + q2

            w_list[:, i, :] = w

        return w_list

#%%

class recurrent_unrolling(nn.Module):
    """
    ablation model: recurrent unrolling

    """

    def __init__(self, num_unroll):
        super(recurrent_unrolling, self).__init__()

        self.layers = num_unroll  # int

        # a fixed value:
        self.gn = nn.Parameter(torch.tensor(0.01), requires_grad=True)
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def prox_log_barrier(self, y, gn, alpha):
        up = y ** 2 + 4 * gn * alpha
        up = torch.clamp(up, 1e-08)
        return (y - torch.sqrt(up)) / 2

    def initialisation(self, m, l, batch_size):
        w = torch.zeros((batch_size, l)).float().to(device)
        v = torch.zeros((batch_size, m)).float().to(device)
        return w, v

    def forward(self, z):
        batch_size, l = z.size()

        z = check_tensor(z, device)
        m = int((1 / 2) * (1 + math.sqrt(1 + 8 * l)))
        D = coo_to_sparseTensor(get_degree_operator(m))

        # initialise:
        w, v = self.initialisation(m, l, batch_size)
        zero_vec = torch.zeros((batch_size, l)).to(device)
        w_list = torch.empty(size=(batch_size, self.layers, l))

        for i in range(self.layers):

            y1 = w - self.gn * (2 * self.beta * w + torch.matmul(v, D))
            y2 = v + self.gn * torch.matmul(w, D.T)

            p1 = torch.max(zero_vec, y1 - 2 * self.gn * z)
            p2 = self.prox_log_barrier(y2, self.gn, self.alpha)

            q1 = p1 - self.gn * (2 * self.beta * p1 + torch.matmul(p2, D))
            q2 = p2 + self.gn * torch.matmul(p1, D.T)

            w = w - y1 + q1
            v = v - y2 + q2

            w_list[:, i, :] = w

        return w_list

#%%
