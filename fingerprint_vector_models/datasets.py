import torch
from torch.utils.data import Dataset, DataLoader

class MyMLPDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        X = self.X[index].float()
        Y = self.Y[index].long()
        return X, Y

class MyTDNNDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        X = self.X[index].unsqueeze(0).float()
        Y = self.Y[index].long()
        return X, Y

def get_mlp_loader(train_data, train_labels, val_data, val_labels, test_data, test_labels, cuda, batch_size):
    if cuda:
        loader_args = dict(batch_size=batch_size, num_workers=16, pin_memory=True)
    else:
        loader_args = dict(batch_size=64)
    train_dataset = MyMLPDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_dataset = MyMLPDataset(val_data, val_labels)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_dataset = MyMLPDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader

def get_tdnn_loader(train_data, train_labels, val_data, val_labels, test_data, test_labels, cuda, batch_size):
    if cuda:
        loader_args = dict(batch_size=batch_size, num_workers=16, pin_memory=True)
    else:
        loader_args = dict(batch_size=64)
    train_dataset = MyTDNNDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_dataset = MyTDNNDataset(val_data, val_labels)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_dataset = MyTDNNDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader
