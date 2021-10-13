

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import math
from src.utils import *


#%%
class ADMM():
    def __init__(self, l2_penalty, log_penalty, step_size=1e-02, relaxation_factor = 1.8):
        self.alpha = log_penalty  # the penalty before log barrier
        self.beta = l2_penalty  # the penalty before l2 term
        self.gn = step_size
        self.relax = relaxation_factor

    def initialisation(self, m, l, batch_size):
        w = torch.zeros((batch_size, l)).float().to(device)
        v = torch.zeros((batch_size, m)).float().to(device)
        return w, v

    def prox_log_barrier(self, y, gn, alpha):
        up = y ** 2 + 4 * gn * alpha
        up = torch.clamp(up, 1e-08)
        return (y - torch.sqrt(up)) / 2

    def objective(self, w, D, z):
        f1 = self.beta * torch.norm(w, 2) ** 2
        f2 = w.T @ z
        f3 = - self.alpha * torch.sum(torch.log(D @ w))

        if all(np.round(w, 4) >= 0):
            return f1 + f2 + f3
        else:
            return 10**3

    def solve(self, z, max_ite=1000, verbose=True):

        batch_size, l = z.size()

        z = check_tensor(z, device)
        m = int((1 / 2) * (1 + math.sqrt(1 + 8 * l)))
        D = coo_to_sparseTensor(get_degree_operator(m))

        # initialise:
        w, v = self.initialisation(m, l, batch_size)
        zero_vec = torch.zeros((batch_size, l)).to(device)
        w_list = torch.empty(size=(batch_size, max_ite, l))
        print(w_list.shape)

        lambda_ = self.relax

        for i in range(max_ite):

            y1 = w - self.gn * (2 * self.beta * w + torch.matmul(v, D))
            p1 = torch.max(zero_vec, y1 - 2 * self.gn * z)

            y2 = v + self.gn * torch.matmul(2 * p1 - w, D.T)
            p2 = self.prox_log_barrier(y2, self.gn, self.alpha)

            w = w + lambda_ * (p1 - w)
            v = v + lambda_ * (p2 - v)

            w_list[:, i, :] = w

        return w_list

#%%

class PDS():
    def __init__(self, l2_psi, log_psi, step_size):
        self.alpha = log_psi  # the penalty before log barrier
        self.beta = l2_psi  # the penalty before l2 term
        self.gn = step_size

    def prox_log_barrier(self, y, gn, alpha):
        return (y - torch.sqrt(y ** 2 + 4 * gn * alpha)) / 2

    def initialisation(self, m, l, batch_size):
        w = torch.zeros((batch_size, l)).float().to(device)
        v = torch.zeros((batch_size, m)).float().to(device)
        return w, v

    def solve(self, z, max_iter = 500):
        batch_size, l = z.size()

        z = check_tensor(z, device)
        m = int((1 / 2) * (1 + math.sqrt(1 + 8 * l)))
        D = coo_to_sparseTensor(get_degree_operator(m)).to(device)

        # initialise:
        w, v = self.initialisation(m, l, batch_size)
        zero_vec = torch.zeros((batch_size, l)).to(device)
        w_list = torch.empty(size=(batch_size, max_iter, l)).to(device)

        for i in range(max_iter):

            y1 = w - self.gn * (2 * self.beta * w + 2 * z + torch.matmul(v, D))
            y2 = v + self.gn * torch.matmul(w, D.T)

            p1 = torch.max(zero_vec, y1)
            p2 = self.prox_log_barrier(y2, self.gn, self.alpha)

            q1 = p1 - self.gn * (2 * self.beta * p1 + 2 * z + torch.matmul(p2, D))
            q2 = p2 + self.gn * torch.matmul(p1, D.T)

            w = w - y1 + q1
            v = v - y2 + q2

            w_list[:, i, :] = w

        return w_list

#%%