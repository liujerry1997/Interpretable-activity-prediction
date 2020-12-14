import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def train(model, train_loader, optimizer, criterion, device, epoch, empty_cache=False):
    start = time.time()
    model.train()
    avg_loss = 0.0
    train_loss = []
    correct = 0
    total = 0

    for batch_num, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y)
        
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item()
        train_loss.extend([loss.item()]*x.size()[0])

        predicted = torch.max(output.data, 1)[1]
        total += y.size(0)
        correct += (predicted == y).sum().item()

        if batch_num % 50 == 49:
            print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}\tElapsed Time: {:.4f}'.
                  format(epoch+1, batch_num+1, avg_loss/50, time.time()-start))
            avg_loss = 0.0    
        
        if empty_cache:
            torch.cuda.empty_cache()
            del x
            del y
            del loss
        
    end = time.time()
    print('Epoch: {}\tTraining Time: {:.4f}'.format(epoch+1, end-start))
    return np.mean(train_loss), correct / total

def valid(model, val_loader, criterion, device, epoch, empty_cache=False):
    start = time.time()
    model.eval()
    val_loss = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_num, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            
            loss = criterion(output, y)
            val_loss.extend([loss.item()]*x.size()[0])

            predicted = torch.max(output.data, 1)[1]
            total += y.size(0)
            correct += (predicted == y).sum().item()

            if empty_cache:
                torch.cuda.empty_cache()
                del x
                del y
                del loss
    
    model.train()
    end = time.time()
    if epoch is None:
        print('Final Validation Time: {:.4f}'.format(end-start))
    else:
        print('Epoch: {}\tValidation Time: {:.4f}'.format(epoch+1, end-start))
    return np.mean(val_loss), correct / total

def test(model, test_loader, device, empty_cache):
    start = time.time()
    model.eval()
    pred_label = []
    true_label = []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            predicted = torch.max(output.data, 1)[1]
            pred_label.extend(predicted.detach().cpu().numpy())
            true_label.extend(y.detach().cpu().numpy())
            
            if empty_cache:
                torch.cuda.empty_cache()
                del x
                del y
    
    true_label = np.array(true_label)
    pred_label = np.array(pred_label)
    accuracy = np.mean(true_label == pred_label)
    auc = roc_auc_score(true_label, pred_label)
    matrix = confusion_matrix(true_label, pred_label)
    end = time.time()
    print('Testing Time: {:.4f}'.format(end-start))
    return accuracy, auc, matrix

def train_model(model, model_name, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs, empty_cache=True):
    train_losses = []
    valid_losses = []
    train_accuracy = []
    valid_accuracy = []
    
    for epoch in range(num_epochs):
        print("Epoch: {}\tLearning Rate: {}".format(epoch+1, optimizer.param_groups[0]['lr']))
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch, empty_cache)
        print('Epoch: {}\tTrain Loss: {:.5f}\tTrain Accuracy: {:.5f}'.format(epoch+1, train_loss, train_acc))
        train_losses.append(train_loss)
        train_accuracy.append(train_acc)
        
        val_loss, val_acc = valid(model, val_loader, criterion, device, epoch, empty_cache)
        print('Epoch: {}\tVal Loss: {:.5f}\tVal Accuracy: {:.5f}'.format(epoch+1, val_loss, val_acc))
        valid_losses.append(val_loss)
        valid_accuracy.append(val_acc)
        
        scheduler.step(val_loss)
    
    return train_losses, train_accuracy, valid_losses, valid_accuracy

def training_plot(a, b, name, graph_type):
    plt.figure(1)
    plt.plot(range(1, len(a)+1), a, 'b', label="train")
    plt.plot(range(1, len(b)+1), b, 'g', label="valid")
    plt.title('Training/Valid {}'.format(graph_type))
    plt.legend()
    plt.savefig('{}_{}_plot.png'.format(name, graph_type.lower()))
    plt.show()
