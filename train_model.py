import torch
import torch.nn.functional as F
from torch import optim, nn

from utils.general_utils import print_progress


def predict(data_loader, device, model):
    model.eval()
    y_true, y_pred = torch.tensor([]).to(device), torch.tensor([]).to(device)
    for batched_graph, labels in data_loader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        logits = model(batched_graph)
        y_true = torch.cat((y_true, labels), 0)
        y_pred = torch.cat((y_pred, logits), 0)
    return F.one_hot(y_true, 2), y_pred


def evaluate(data_loader, device, model):
    model.eval()
    total = 0
    total_correct = 0
    for batched_graph, labels in data_loader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        total += len(labels)
        logits = model(batched_graph)
        _, predicted = torch.max(logits, 1)
        total_correct += (predicted == labels).sum().item()
    acc = 1.0 * total_correct / total
    return acc


def train(device, model, save_path, train_loader, val_loader, max_epoch=300):
    # loss function, optimizer and scheduler
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5)
    model.to(device)
    # training loop
    loss_history = []
    valid_curve = []
    last_record_val_acc = 0
    for epoch in range(max_epoch):
        model.train()
        total_loss = 0
        for batch, (batched_graph, labels) in enumerate(train_loader):
            print_progress(batch, len(train_loader), prefix=f'Epoch {epoch}: ')
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            logits = model(batched_graph)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        acc = evaluate(train_loader, device, model)
        val_acc = evaluate(val_loader, device, model)
        valid_curve.append(val_acc)
        print()
        print(f'Epoch {epoch} | Loss {total_loss / (batch + 1):8.2f} | acc. {acc:8.3f} | val_acc. {val_acc:8.3f} ')
        if val_acc > last_record_val_acc or epoch == 0:
            print(f'val_acc increase : {last_record_val_acc} --> {val_acc}, save model.')
            last_record_val_acc = val_acc
            # save the best model guided by val_mae
            torch.save(model, save_path)
    return loss_history, valid_curve
