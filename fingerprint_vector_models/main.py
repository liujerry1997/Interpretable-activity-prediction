import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from preprocessing import convert_smiles_to_numpy
from datasets import MyMLPDataset, MyTDNNDataset, get_mlp_loader, get_tdnn_loader
from training import valid, test, train_model, training_plot

def read_data():
    training_set = pd.read_csv('interpretable-activity-prediction/data/training_set.csv')
    training_set = training_set.loc[:,['SMILES', 'Activity']]
    testing_set = pd.read_csv('interpretable-activity-prediction/data/test_set_filtered.csv')
    testing_set = testing_set.loc[:,['SMILES', 'Activity']]

    radius = 2
    num_bits = 2048

    training_x = np.array([convert_smiles_to_numpy(smiles, radius, num_bits) for smiles in training_set['SMILES']])
    x_test = np.array([convert_smiles_to_numpy(smiles, radius, num_bits) for smiles in testing_set['SMILES']])
    training_y = ((training_set['Activity'] == 'Active') * 1.0).to_numpy()
    y_test = ((testing_set['Activity'] == 'Active') * 1.0).to_numpy()

    x_train, x_val, y_train, y_val = train_test_split(training_x, training_y, test_size=0.2, random_state=23)

    print('Active train:', np.sum(y_train == 1))
    print('Inactive train:', np.sum(y_train == 0))
    print('Active val:', np.sum(y_val == 1))
    print('Inactive val:', np.sum(y_val == 0))
    print('Active test:', np.sum(y_test == 1))
    print('Inactive test:', np.sum(y_test == 0))

    return x_train, x_val, y_train, y_val, x_test, y_test

def linear_classifier():
    linear = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    linear.fit(x_train, y_train)
    train_predict = linear.predict(x_train)
    train_acc = np.mean(train_predict == y_train)
    print('Train Accuracy: {:.5f}'.format(train_acc))
    val_predict = linear.predict(x_val)
    val_acc = np.mean(val_predict == y_val)
    print('Val Accuracy: {:.5f}'.format(val_acc))
    test_predict = linear.predict(x_test)
    test_acc = np.mean(test_predict == y_test)
    print('Test Accuracy: {:.5f}'.format(test_acc))
    auc = roc_auc_score(y_test, test_predict)
    print('AUC Score: {:.5f}'.format(auc))

def svm_classifier():
    svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    svm.fit(x_train, y_train)
    train_predict = svm.predict(x_train)
    train_acc = np.mean(train_predict == y_train)
    print('Train Accuracy: {:.5f}'.format(train_acc))
    val_predict = svm.predict(x_val)
    val_acc = np.mean(val_predict == y_val)
    print('Val Accuracy: {:.5f}'.format(val_acc))
    test_predict = svm.predict(x_test)
    test_acc = np.mean(test_predict == y_test)
    print('Test Accuracy: {:.5f}'.format(test_acc))
    auc = roc_auc_score(y_test, test_predict)
    print('AUC Score: {:.5f}'.format(auc))

def mlp_classifier():
    mlp = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(128),
        nn.Linear(128, 2)
    )
    mlp.to(device)
    print(mlp)

    num_epochs = 100

    model_name = 'Simple_MLP'

    lr = 1e-3
    weight_decay = 5e-6

    start_epoch = 0
    train_losses = []
    valid_losses = []

    empty_cache = True

    batch_size = 64
    train_loader, val_loader, test_loader = get_mlp_loader(x_train, y_train, x_val, y_val, x_test, y_test, cuda, batch_size)

    optimizer = optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=1, verbose=True)
    criterion = nn.CrossEntropyLoss()

    train_losses, valid_losses = train_model(mlp, model_name, train_loader, val_loader, optimizer, criterion, scheduler, device, start_epoch, num_epochs, train_losses, valid_losses, empty_cache)

    training_plot(train_losses, valid_losses)
    
    test_loss, test_acc = valid(mlp, test_loader, criterion, device, None, empty_cache)
    print('Test Loss: {:.5f}\tTest Accuracy: {:.5f}'.format(test_loss, test_acc))

    test_predict = test(mlp, test_loader, device, empty_cache)
    auc = roc_auc_score(y_test, test_predict)
    print('AUC Score: {:.5f}'.format(auc))

def tdnn_with_padding_classifier():
    tdnn = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(128, 2)
    )
    tdnn.to(device)
    print(tdnn)

    num_epochs = 100

    model_name = 'Simple_TDNN'

    lr = 1e-3
    weight_decay = 5e-6

    start_epoch = 0
    train_losses = []
    valid_losses = []

    empty_cache = True

    batch_size = 64
    train_loader, val_loader, test_loader = get_tdnn_loader(x_train, y_train, x_val, y_val, x_test, y_test, cuda, batch_size)

    optimizer = optim.Adam(tdnn.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=1, verbose=True)
    criterion = nn.CrossEntropyLoss()

    train_losses, valid_losses = train_model(tdnn, model_name, train_loader, val_loader, optimizer, criterion, scheduler, device, start_epoch, num_epochs, train_losses, valid_losses, empty_cache)

    training_plot(train_losses, valid_losses)

    test_loss, test_acc = valid(tdnn, test_loader, criterion, device, None, empty_cache)
    print('Test Loss: {:.5f}\tTest Accuracy: {:.5f}'.format(test_loss, test_acc))

    test_predict = test(tdnn, test_loader, device, empty_cache)
    auc = roc_auc_score(y_test, test_predict)
    print('AUC Score: {:.5f}'.format(auc))

def tdnn_without_padding_classifier():
    tdnn = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(128, 2)
    )
    tdnn.to(device)
    print(tdnn)

    num_epochs = 100

    model_name = 'Simple_TDNN'

    lr = 0.15
    weight_decay = 5e-5
    momentum = 0.9

    start_epoch = 0
    train_losses = []
    valid_losses = []

    empty_cache = True

    batch_size = 64
    train_loader, val_loader, test_loader = get_tdnn_loader(x_train, y_train, x_val, y_val, x_test, y_test, cuda, batch_size)

    optimizer = optim.SGD(tdnn.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=1, verbose=True)
    criterion = nn.CrossEntropyLoss()

    train_losses, valid_losses = train_model(tdnn, model_name, train_loader, val_loader, optimizer, criterion, scheduler, device, start_epoch, num_epochs, train_losses, valid_losses, empty_cache)

    training_plot(train_losses, valid_losses)

    test_loss, test_acc = valid(tdnn, test_loader, criterion, device, None, empty_cache)
    print('Test Loss: {:.5f}\tTest Accuracy: {:.5f}'.format(test_loss, test_acc))
    
    test_predict = test(tdnn, test_loader, device, empty_cache)
    auc = roc_auc_score(y_test, test_predict)
    print('AUC Score: {:.5f}'.format(auc))

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(device)

    x_train, x_val, y_train, y_val, x_test, y_test = read_data()

    linear_classifier()
    svm_classifier()
    mlp_classifier()
    tdnn_with_padding_classifier()
    tdnn_without_padding_classifier()
