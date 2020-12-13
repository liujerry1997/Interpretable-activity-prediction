import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def train(models, train_dataloaders, val_dataloaders, optimizers, schedulers, criterion, num_epoch, k, ensemble):
    train_losses_list, val_losses_list, train_accs_list, val_accs_list, auc_scores_list = [], [], [], [], []
    for i in range(ensemble):
        print("training ensemble model", i + 1)

        model = models[i]
        train_dataloader = train_dataloaders[i]
        val_dataloader = val_dataloaders[i]
        optimizer = optimizers[i]
        scheduler = schedulers[i]

        true_labels = np.array([])
        probs = np.array([])

        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        for epoch in range(num_epoch):
            model.train()
            correct = 0
            total = 0
            running_loss = 0
            for batch_idx, (X, Y) in enumerate(train_dataloader):
                optimizer.zero_grad()
                output = model.forward(X)
                probs = np.append(probs, output[:, -1].cpu().detach().numpy())
                true_labels = np.append(true_labels, Y.cpu().detach().numpy())

                pred = torch.argmax(output, dim=1)

                correct += (pred == Y).sum().item()
                total += Y.size(0)
                train_acc = correct / total
                loss = criterion(output, Y)
                running_loss += loss.item() * Y.size(0)
                train_loss = running_loss / total
                loss.backward()
                optimizer.step()

            train_accs.append(train_acc)
            train_losses.append(train_loss)
            val_acc, val_loss, _ = validate(model, val_dataloader, criterion, k)
            scheduler.step(val_loss)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            auc_scores = roc_auc_score(true_labels, probs)

        train_losses_list.append(train_losses)
        val_losses_list.append(val_losses)
        train_accs_list.append(train_accs)
        val_accs_list.append(val_accs)
        auc_scores_list.append(auc_scores)

    # print(train_losses_list, val_losses_list)
    if ensemble == 1:
        return train_losses_list[0], val_losses_list[0], train_accs_list[0], val_accs_list[0], auc_scores_list[0]

    else:
        train_ave_loss = np.average(np.array(train_losses_list), axis=0)
        val_ave_loss = np.average(np.array(val_losses_list), axis=0)
        train_ave_acc = np.average(np.array(train_accs_list), axis=0)
        val_ave_acc = np.average(np.array(val_accs_list), axis=0)

        return train_ave_loss, val_ave_loss, train_ave_acc, val_ave_acc, auc_scores_list


def validate(model, dataloader, criterion, k):
    model.eval()
    num_correct = 0
    total = 0
    running_loss = 0

    true_labels = np.array([])
    probs = np.array([])

    preds = []
    for batch_idx, (X, Y) in enumerate(dataloader):
        output = model.forward(X)
        pred = torch.argmax(output, dim=1)
        # preds.append(pred)
        num_correct += (pred == Y).sum().item()
        total += Y.size(0)
        loss = criterion(output, Y)
        running_loss += loss.item() * Y.size(0)
        # print("preds", preds)
        probs = np.append(probs, output[:, -1].cpu().detach().numpy())
        true_labels = np.append(true_labels, Y.cpu().detach().numpy())

    auc_scores = roc_auc_score(true_labels, probs)

    return num_correct / total, running_loss / total, auc_scores


def validate_conf_matrix(model, dataloader, criterion, k):
    model.eval()
    num_correct = 0
    total = 0
    running_loss = 0

    true_labels = np.array([])
    probs = np.array([])

    preds = np.array([])
    for batch_idx, (X, Y) in enumerate(dataloader):
        output = model.forward(X)
        pred = torch.argmax(output, dim=1)
        preds = np.append(preds, pred.cpu().detach().numpy())
        num_correct += (pred == Y).sum().item()
        total += Y.size(0)
        loss = criterion(output, Y)
        running_loss += loss.item() * Y.size(0)
        # print("preds", preds)
        probs = np.append(probs, output[:, -1].cpu().detach().numpy())
        true_labels = np.append(true_labels, Y.cpu().detach().numpy())

    auc_scores = roc_auc_score(true_labels, probs)

    return num_correct / total, running_loss / total, auc_scores, true_labels, preds

