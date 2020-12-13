import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.sparse as sp

def normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()





def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sgc_precompute(features, adj, degree):
    adj = normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    features = torch.tensor(features)
    for i in range(degree):
        features = torch.spmm(adj, features)
    return features   # aton_num * 133



class MyDataset(Dataset):
    def __init__(self, X, A, Y, degree):
        self.X = X
        self.A = A
        self.Y = Y
        self.degree = degree
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, index):
        X = sgc_precompute(self.X[index], self.A[index], self.degree)
        return torch.sum(X, dim = 0, keepdim = False), torch.tensor(self.Y[index])


class MyDataset_mol(Dataset):   # molecular feature added
    def __init__(self, X, A, Y, degree, mol_feat):
        self.X = X
        self.A = A
        self.Y = Y
        self.degree = degree
        self.mol_feat = mol_feat

    def __len__(self):
        return len(self.Y)
    def __getitem__(self, index):
        X = sgc_precompute(self.X[index], self.A[index], self.degree)
        sum_atom_feat = torch.sum(X, dim = 0)
        all_feat = np.concatenate( (self.mol_feat[index] , sum_atom_feat), axis=0)

        return torch.tensor(all_feat).float(), torch.tensor(self.Y[index])
