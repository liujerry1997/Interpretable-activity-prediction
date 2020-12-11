import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np

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
    predicted_test = []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            output = model(x)
            predicted = torch.max(output.data, 1)[1]
            predicted_test.extend(predicted.cpu().detach().numpy())
            
            if empty_cache:
                torch.cuda.empty_cache()
                del x
    
    end = time.time()
    print('Testing Time: {:.4f}'.format(end-start))
    return predicted_test

def train_model(model, model_name, train_loader, val_loader, optimizer, criterion, scheduler, device, start_epoch, num_epochs, train_losses, valid_losses, empty_cache=False):
    for epoch in range(start_epoch, num_epochs):
        print("Epoch: {}\tLearning Rate: {}".format(epoch+1, optimizer.param_groups[0]['lr']))
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch, empty_cache)
        print('Epoch: {}\tTrain Loss: {:.5f}\tTrain Accuracy: {:.5f}'.format(epoch+1, train_loss, train_acc))
        train_losses.append(train_loss)
        
        val_loss, val_acc = valid(model, val_loader, criterion, device, epoch, empty_cache)
        print('Epoch: {}\tVal Loss: {:.5f}\tVal Accuracy: {:.5f}'.format(epoch+1, val_loss, val_acc))
        valid_losses.append(val_loss)
        
        scheduler.step(val_loss)
    
    return train_losses, valid_losses

def training_plot(a, b):
    plt.figure(1)
    plt.plot(a, 'b', label="train")
    plt.plot(b, 'g', label="valid")
    plt.title('Training/Valid Loss')
    plt.legend()
    plt.savefig('Loss Plot.png')
    plt.show()
