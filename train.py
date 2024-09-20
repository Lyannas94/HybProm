import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
import math
import random
from collections import Counter
import model

def train_model(dataset, model, learning_rate=1e-4, batch_size=128, epoch_n=200, random_seed=200,
                val_split=0.1, weighted_sampling=False, model_name='D_noTA.pt',
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # Random seed
    random.seed(random_seed)

    # Load data
    data_list = list(range(0, len(dataset)))
    val_list = random.sample(data_list, int(len(dataset) * val_split))
    trainset = [dataset[i] for i in data_list if i not in val_list]
    valset = [dataset[i] for i in data_list if i in val_list]
    # Weighed Sampling
    if weighted_sampling:
        label_count = Counter([int(data[1]) for data in dataset])
        weights = [100 / label_count[int(data[1])] for data in trainset]
        sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)
        train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler, drop_last=True)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, drop_last=True)

    # optimizer and L2 regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    model.to(device)
    # 训练循环
    for epoch in range(epoch_n):
        model.train()
        running_loss = 0.0
        train_acc = 0.0
        old_train_acc = 0
        for i, batch in enumerate(train_loader):
            batch = [item.to(device) for item in batch]
            labels = batch[1]
            # forward + backprop + loss
            pred = model(batch)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            # update model params
            optimizer.step()
            running_loss += loss.detach().item()
            train_acc += (torch.argmax(pred, 1).flatten() == labels).type(torch.float).mean().item()
        val_acc = evaluation(val_loader, model, device)

        if train_acc > old_train_acc:
            old_train_acc = train_acc
            torch.save(model, model_name)
        print("Epoch {}| Loss: {:.4f}| Train accuracy: {:.4f}| Validation accuracy: {:.4f}".
              format(epoch, running_loss / (i + 1), train_acc / (i + 1), val_acc))


def test_model(dataset, model_name='E.pt', val_split=0.5,
               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    data_list = list(range(0, len(dataset)))
    test_list = random.sample(data_list, int(len(dataset) * val_split))
    testset = [dataset[i] for i in data_list if i in test_list]
    test_loader = DataLoader(dataset, batch_size=len(testset), shuffle=True, drop_last=True)
    model = torch.load(model_name, map_location=device)
    model.eval()
    model.to(device)

    TP, FN, FP, TN = 0, 0, 0, 0
    for data in test_loader:
        with torch.no_grad():
            inputs = data[0].to(device), data[1].to(device)
            labels = data[1].to(device)
            pred = model(inputs)
            pred = pred.argmax(dim=1)
            A, B, C, D = eff(labels, pred)
            TP += A
            FN += B
            FP += C
            TN += D
    SN, SP, ACC, MCC, F1Score, PRE, Err = Judeff(TP, FN, FP, TN)

    print("TP: {}, FN: {}, FP: {}, TN: {}".format(TP, FN, FP, TN))
    print("SN: {:.4f}, SP: {:.4f}, ACC: {:.4f}, MCC: {:.4f}, F1Score: {:.4f}, PRE: {:.4f}, Err: {:.4f}".format(SN, SP,
                                                                                                               ACC, MCC,
                                                                                                               F1Score,
                                                                                                               PRE,
                                                                                                               Err))
    print("{:.4f},{:.4f}, {:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(SN, SP, ACC, MCC, F1Score, PRE, Err))


def evaluation(loader, model, device=torch.device('cuda')):
    model.eval()
    model.to(device)
    correct = 0
    total_loss = 0.0
    total = len(loader.dataset)
    for data in loader:
        with torch.no_grad():
            inputs = data[0].to(device), data[1].to(device)
            labels = data[1].to(device)
            pred = model(inputs)
            pred = pred.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
    acc = correct / total
    return acc


def eff(labels, preds):
    TP, FN, FP, TN = 0, 0, 0, 0
    for idx, label in enumerate(labels):

        if label == 1:
            if label == preds[idx]:
                TP += 1
            else:
                FN += 1
        elif label == preds[idx]:
            TN += 1
        else:
            FP += 1

    return TP, FN, FP, TN


def Judeff(TP, FN, FP, TN):
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + FN + FP + TN)
    MCC = (TP * TN - FP * FN) / (math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))
    F1Score = (2 * TP) / (2 * TP + FN + FP)
    PRE = TP / (TP + FP)
    Err = 1 - ((TP + TN) / (TP + FN + FP + TN))

    return SN, SP, ACC, MCC, F1Score, PRE, Err