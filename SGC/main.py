import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn import model_selection
from training import *
from dataset import *
import torch.optim as optim

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        return self.W(x)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight)


def split_pos_neg(X_set, Y):
    pos_X_set = []
    neg_X_set = []
    for i in range(len(X_set)):
        if Y[i] == 1:
            X_set_Y = np.append(X_set[i], [1])
            # print(X_set_Y.shape)
            pos_X_set.append(X_set_Y)
        else:
            X_set_Y = np.append(X_set[i], [0])
            neg_X_set.append(X_set_Y)
    return np.array(pos_X_set), np.array(neg_X_set)




if __name__ == '__main__':

    ##############
    ensemble = 20
    ##############

    # n_feature = 133 + 200
    n_feature = 133
    n_class = 2
    k = 6
    num_epoch = 30
    lr = 0.001
    wd = 5e-6

    train_f = np.load("data/train_feature.npy", allow_pickle=True)
    train_adj_mx = np.load("data/train_adjacent_matrix.npy", allow_pickle=True)
    train_mol_f = np.load("data/train_mol_features.npy", allow_pickle=True)
    train_label = np.load("data/train_label.npy", allow_pickle=True)

    SGC_base_list = []
    optimizer_list = []
    scheduler_list = []

    for i in range(ensemble):
        SGC_base = SGC(n_feature, n_class)
        SGC_base.apply(init_weights)
        SGC_base_list.append(SGC_base)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(SGC_base.parameters(), lr = lr, weight_decay = wd)
        optimizer_list.append(optimizer)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience = 0)
        scheduler_list.append(scheduler)

    concat_f = []
    for count in range(train_f.shape[0]):  # 2335
        one_mol_f = []
        one_mol_f.extend([train_f[count], train_adj_mx[count], train_mol_f[count]])
        concat_f.append(one_mol_f)

    concat_f = np.asarray(concat_f)
    train_X_set, val_X_set, train_Y, val_Y = model_selection.train_test_split(concat_f, train_label, test_size=0.2,
                                                                              random_state=42)

    pos_X_set, neg_X_set = split_pos_neg(train_X_set, train_Y)  # Take all

    for i in range(ensemble):
        # For each ensemble, pick all positive samples and same number of negative samples. (balanced 1:1)
        mol_number = neg_X_set.shape[0]
        random_indices = np.random.choice(mol_number, size=len(pos_X_set) * 2, replace=False)
        part_neg_X_set = neg_X_set[random_indices]

        train_X_set = np.concatenate((pos_X_set, part_neg_X_set))
        # print(train_X_set[:, 3])

        np.random.shuffle(train_X_set)
        # print(train_X_set[:, 3])

        train_X = train_X_set[:, 0]
        train_A = train_X_set[:, 1]
        train_mol_X = train_X_set[:, 2]
        train_Y = train_X_set[:, 3]  # This partial train_Y overwrites the overall train_Y before.

        val_X = val_X_set[:, 0]
        val_A = val_X_set[:, 1]
        val_mol_f = val_X_set[:, 2]

        degree = 8

        train_dataset = MyDataset_mol(train_X, train_A, train_Y, degree, train_mol_X)
        val_dataset = MyDataset_mol(val_X, val_A, val_Y, degree, val_mol_f)

        train_dataloader_args = dict(shuffle=True, batch_size=8, drop_last=True)
        val_dataloader_args = dict(shuffle=False, batch_size=8, drop_last=True)
        train_dataloader = DataLoader(train_dataset, **train_dataloader_args)
        val_dataloader = DataLoader(val_dataset, **val_dataloader_args)

        train_dataloader_list.append(train_dataloader)
        val_dataloader_list.append(val_dataloader)

        train_losses, val_losses, train_accs, val_accs, auc_scores = train(SGC_base_list, train_dataloader_list,
                                                                           val_dataloader_list,
                                                                           optimizer_list, scheduler_list, criterion,
                                                                           num_epoch, k, ensemble)


