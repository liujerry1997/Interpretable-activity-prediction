import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import seaborn as sns
import matplotlib.pyplot as plt

from preprocessing import convert_smiles_to_numpy
from datasets import MyMLPDataset, MyTDNNDataset, get_mlp_loader, get_tdnn_loader
from training import valid, test, train_model, training_plot

def read_data():
    training_set = pd.read_csv('interpretable-activity-prediction/data/training_set.csv')
    training_set = training_set.loc[:,['SMILES', 'Activity']]
    testing_set = pd.read_csv('interpretable-activity-prediction/data/test_set_filtered.csv')
    testing_set = testing_set.loc[:,['SMILES', 'Activity']]

    print(training_set.head())
    print(testing_set.head())

    radius = 2
    num_bits = 2048

    training_x = np.array([convert_smiles_to_numpy(smiles, radius, num_bits) for smiles in training_set['SMILES']])
    x_test = np.array([convert_smiles_to_numpy(smiles, radius, num_bits) for smiles in testing_set['SMILES']])
    training_y = ((training_set['Activity'] == 'Active') * 1.0).to_numpy()
    y_test = ((testing_set['Activity'] == 'Active') * 1.0).to_numpy()

    print('Pre-SMOTE Active:', np.sum(training_y == 1))
    print('Pre-SMOTE Inactive:', np.sum(training_y == 0))
    print('Active:')
    print(training_x[training_y == 1])
    print(training_y[training_y == 1])
    print('Inactive:')
    print(training_x[training_y == 0])
    print(training_y[training_y == 0])

    oversampling = SMOTE()
    training_x, training_y = oversampling.fit_resample(training_x, training_y)

    print('Post-SMOTE Active:', np.sum(training_y == 1))
    print('Post-SMOTE Inactive:', np.sum(training_y == 0))
    print('Active:')
    print(training_x[training_y == 1])
    print(training_y[training_y == 1])
    print('Inactive:')
    print(training_x[training_y == 0])
    print(training_y[training_y == 0])

    x_train, x_val, y_train, y_val = train_test_split(training_x, training_y, test_size=0.2, random_state=23)

    print('Active train:', np.sum(y_train == 1))
    print('Inactive train:', np.sum(y_train == 0))
    print('Active val:', np.sum(y_val == 1))
    print('Inactive val:', np.sum(y_val == 0))
    print('Active test:', np.sum(y_test == 1))
    print('Inactive test:', np.sum(y_test == 0))

    return x_train, x_val, y_train, y_val, x_test, y_test

def linear_classifier():
    linear = make_pipeline(StandardScaler(), SGDClassifier(class_weight='balanced'))
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
    stats['Linear Classifier'] = [train_acc, val_acc, test_acc, auc]

    matrix = confusion_matrix(y_test, test_predict)
    matrix_df = pd.DataFrame(matrix, index=['True_Inactive', 'True_Active'], columns=['Pred_Inactive', 'Pred_Active'])
    ax=sns.heatmap(matrix_df, cmap='YlGnBu', annot=True, fmt='d', annot_kws={'size': 16}).set_title('Linear Confusion Matrix (SMOTE)')
    plt.savefig('linear_smote_matrix.png')
    plt.show()

def svm_classifier():
    svm = make_pipeline(StandardScaler(), SVC(class_weight='balanced'))
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
    stats['SVM Classifier'] = [train_acc, val_acc, test_acc, auc]

    matrix = confusion_matrix(y_test, test_predict)
    matrix_df = pd.DataFrame(matrix, index=['True_Inactive', 'True_Active'], columns=['Pred_Inactive', 'Pred_Active'])
    ax=sns.heatmap(matrix_df, cmap='YlGnBu', annot=True, fmt='d', annot_kws={'size': 16}).set_title('SVM Confusion Matrix (SMOTE)')
    plt.savefig('svm_smote_matrix.png')
    plt.show()

def mlp_classifier():
    mlp = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(128, 2)
    )
    mlp.to(device)
    print(mlp)

    num_epochs = 100

    lr = 1e-3
    weight_decay = 5e-6
    momentum = 0.9

    empty_cache = True

    batch_size = 32
    train_loader, val_loader, test_loader = get_mlp_loader(x_train, y_train, x_val, y_val, x_test, y_test, cuda, batch_size)

    optimizer = optim.SGD(mlp.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=1, verbose=True)
    criterion = nn.CrossEntropyLoss()

    model_name = 'mlp_smote_{}_epochs_{}_lr_{}_wd'.format(num_epochs, lr, weight_decay)
    print('Training {}'.format(model_name))
    train_losses, train_accuracy, valid_losses, valid_accuracy = train_model(mlp, model_name, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs, empty_cache)

    training_plot(train_losses, valid_losses, model_name, 'Loss')
    training_plot(train_accuracy, valid_accuracy, model_name, 'Accuracy')
    
    test_loss, test_acc = valid(mlp, test_loader, criterion, device, None, empty_cache)
    print('Test Loss: {:.5f}\tTest Accuracy: {:.5f}'.format(test_loss, test_acc))

    accuracy, auc, matrix = test(mlp, test_loader, device, empty_cache)
    print('Accuracy: {:.5f}'.format(accuracy))
    print('AUC Score: {:.5f}'.format(auc))
    stats['MLP Classifier'] = [train_accuracy[-1], valid_accuracy[-1], test_acc, auc]

    matrix_df = pd.DataFrame(matrix, index=['True_Inactive', 'True_Active'], columns=['Pred_Inactive', 'Pred_Active'])
    ax=sns.heatmap(matrix_df, cmap='YlGnBu', annot=True, fmt='d', annot_kws={'size': 16}).set_title('MLP Confusion Matrix (SMOTE)')
    plt.savefig('{}_matrix.png'.format(model_name))
    plt.show()

def tdnn_with_padding_classifier():
    tdnn1 = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=2048, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm1d(2048),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.MaxPool1d(2),
        nn.Flatten(),
        nn.Linear(128*32, 2)
    )
    tdnn1.to(device)
    print(tdnn1)

    num_epochs = 100

    lr = 1e-3
    weight_decay = 5e-6
    momentum = 0.9

    start_epoch = 0
    train_losses = []
    valid_losses = []

    empty_cache = True

    batch_size = 32
    train_loader, val_loader, test_loader = get_tdnn_loader(x_train, y_train, x_val, y_val, x_test, y_test, cuda, batch_size)

    optimizer = optim.SGD(tdnn1.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=1, verbose=True)
    criterion = nn.CrossEntropyLoss()

    model_name = 'tdnn_with_padding_smote_{}_epochs_{}_lr_{}_wd'.format(num_epochs, lr, weight_decay)
    print('Training {}'.format(model_name))
    train_losses, train_accuracy, valid_losses, valid_accuracy = train_model(tdnn1, model_name, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs, empty_cache)

    training_plot(train_losses, valid_losses, model_name, 'Loss')
    training_plot(train_accuracy, valid_accuracy, model_name, 'Accuracy')

    test_loss, test_acc = valid(tdnn1, test_loader, criterion, device, None, empty_cache)
    print('Test Loss: {:.5f}\tTest Accuracy: {:.5f}'.format(test_loss, test_acc))

    accuracy, auc, matrix = test(tdnn1, test_loader, device, empty_cache)
    print('Accuracy: {:.5f}'.format(accuracy))
    print('AUC Score: {:.5f}'.format(auc))
    stats['TDNN Classifier with Padding'] = [train_accuracy[-1], valid_accuracy[-1], test_acc, auc]

    matrix_df = pd.DataFrame(matrix, index=['True_Inactive', 'True_Active'], columns=['Pred_Inactive', 'Pred_Active'])
    ax=sns.heatmap(matrix_df, cmap='YlGnBu', annot=True, fmt='d', annot_kws={'size': 16}).set_title('TDNN with Padding Confusion Matrix (SMOTE)')
    plt.savefig('{}_matrix.png'.format(model_name))
    plt.show()

def tdnn_without_padding_classifier():
    tdnn2 = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=2048, kernel_size=2, stride=2),
        nn.BatchNorm1d(2048),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=2, stride=2),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=2, stride=2),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv1d(in_channels=512, out_channels=256, kernel_size=2, stride=2),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv1d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.MaxPool1d(2),
        nn.Flatten(),
        nn.Linear(128*32, 2)
    )
    tdnn2.to(device)
    print(tdnn2)

    num_epochs = 100

    lr = 1e-3
    weight_decay = 5e-6
    momentum = 0.9

    start_epoch = 0
    train_losses = []
    valid_losses = []

    empty_cache = True

    batch_size = 32
    train_loader, val_loader, test_loader = get_tdnn_loader(x_train, y_train, x_val, y_val, x_test, y_test, cuda, batch_size)

    optimizer = optim.SGD(tdnn2.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=1, verbose=True)
    criterion = nn.CrossEntropyLoss()

    model_name = 'tdnn_without_padding_smote_{}_epochs_{}_lr_{}_wd'.format(num_epochs, lr, weight_decay)
    print('Training {}'.format(model_name))
    train_losses, train_accuracy, valid_losses, valid_accuracy = train_model(tdnn2, model_name, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs, empty_cache)

    training_plot(train_losses, valid_losses, model_name, 'Loss')
    training_plot(train_accuracy, valid_accuracy, model_name, 'Accuracy')

    test_loss, test_acc = valid(tdnn2, test_loader, criterion, device, None, empty_cache)
    print('Test Loss: {:.5f}\tTest Accuracy: {:.5f}'.format(test_loss, test_acc))
    
    accuracy, auc, matrix = test(tdnn2, test_loader, device, empty_cache)
    print('Accuracy: {:.5f}'.format(accuracy))
    print('AUC Score: {:.5f}'.format(auc))
    stats['TDNN Classifier without Padding'] = [train_accuracy[-1], valid_accuracy[-1], test_acc, auc]

    matrix_df = pd.DataFrame(matrix, index=['True_Inactive', 'True_Active'], columns=['Pred_Inactive', 'Pred_Active'])
    ax=sns.heatmap(matrix_df, cmap='YlGnBu', annot=True, fmt='d', annot_kws={'size': 16}).set_title('TDNN without Padding Confusion Matrix (SMOTE)')
    plt.savefig('{}_matrix.png'.format(model_name))
    plt.show()

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(device)

    x_train, x_val, y_train, y_val, x_test, y_test = read_data()

    stats = {}

    linear_classifier()
    svm_classifier()
    mlp_classifier()
    tdnn_with_padding_classifier()
    tdnn_without_padding_classifier()

    stats_df = pd.DataFrame(stats, index=['Training Accuracy', 'Validation Accuracy', 'Testing Accuracy', 'AUC Score'])
    stats_df.to_csv('smote_result.csv')
    print(stats_df)
