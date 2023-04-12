import torch
import torch.nn as nn
from torch.utils import data
from torch import optim


def train(training_data, validation_data, model, epochs, batch_size, lr, lr_decay, weight_decay):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=lr_decay)
    criterion = MultipleCrossEntropyLoss('mean')
    train_data = data.TensorDataset(*training_data)
    train_loaded = data.DataLoader(dataset=train_data, batch_size=batch_size)

    loss_over_epoch = []
    total_acc = []
    for epoch in range(epochs):
        print(f'\n------------------------ epoch = {epoch + 1} -----------------------------')
        model.train()
        for i, x_batch in enumerate(train_loaded):
            print(f'--------------------- processing batch no. {i + 1} -------------------------')
            # x_batch=x_batch.cuda()
            x = x_batch[:-1]
            y = x_batch[-1]

            y_pred = model(*x)
            loss = criterion(y_pred, y)
            loss_over_epoch.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation data
        x = validation_data[:-1]
        y_test = validation_data[-1]

        y_hat = predict(model, x, batch_size)

        if torch.max(y_test) > 3:
            acc, _ = evaluate(y_test, y_hat, 9)
        else:
            acc, _ = evaluate(y_test, y_hat, 4)
        total_acc.append(acc)

    return loss_over_epoch, total_acc


def predict(model, x_test, batch_size):
    """
    Output <- Model(Input)
    """
    model.eval()
    x = data.TensorDataset(*x_test)
    x = data.DataLoader(dataset=x, batch_size=batch_size)

    output = [model(*x_batch) for x_batch in x]

    return torch.cat(output, dim=0)


def evaluate(y_true, y, q):
    struc = range(q)
    y_pred = torch.argmax(y, dim=2)

    total_protein_correct_count = torch.zeros(q - 1)

    for i in range(len(y_true)):
        total_protein_correct_count = total_protein_correct_count + help((y_pred[i]), (y_true[i]), q)

    _, true_count = y_true.unique(return_counts=True)
    true_count = true_count[1:]

    acc_per_struc = total_protein_correct_count / true_count

    total_accuracy = sum(total_protein_correct_count) / sum(true_count)

    for number, i in enumerate(acc_per_struc):
        print(f'Accuracy of structure {number + 1} = {i}')

    print(f'--\nThe total accuracy = {total_accuracy}')

    print(f'--\nPercentage of each correctly predicted structure with respect to total correctly predicted structures:')
    for number, i in enumerate(total_protein_correct_count):
        print(f'Structure {number + 1} = {i / sum(total_protein_correct_count)}')

    return total_accuracy, acc_per_struc


def help(y_pred, y_true, q):
    # total_correct = sum([1 for i in range(len(y_pred)) if y_pred[i] == y_true[i]])

    if q == 4:
        correctH = sum([1 for i in range(len(y_pred)) if y_pred[i] == 1 and y_true[i] == 1])
        correctB = sum([1 for i in range(len(y_pred)) if y_pred[i] == 2 and y_true[i] == 2])
        correctE = sum([1 for i in range(len(y_pred)) if y_pred[i] == 3 and y_true[i] == 3])

        return torch.as_tensor([correctH, correctB, correctE])

    if q == 9:
        correctL = sum([1 for i in range(len(y_pred)) if y_pred[i] == 1 and y_true[i] == 1])
        correctB = sum([1 for i in range(len(y_pred)) if y_pred[i] == 2 and y_true[i] == 2])
        correctE = sum([1 for i in range(len(y_pred)) if y_pred[i] == 3 and y_true[i] == 3])
        correctG = sum([1 for i in range(len(y_pred)) if y_pred[i] == 4 and y_true[i] == 4])
        correctI = sum([1 for i in range(len(y_pred)) if y_pred[i] == 5 and y_true[i] == 5])
        correctH = sum([1 for i in range(len(y_pred)) if y_pred[i] == 6 and y_true[i] == 6])
        correctS = sum([1 for i in range(len(y_pred)) if y_pred[i] == 7 and y_true[i] == 7])
        correctT = sum([1 for i in range(len(y_pred)) if y_pred[i] == 8 and y_true[i] == 8])

        return torch.as_tensor([correctL, correctB, correctE, correctG, correctI, correctH, correctS, correctT])


class MultipleCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        y_true = y_true.type(torch.LongTensor)
        mask = y_true > 0
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        if torch.max(y_true) == 3:
            weights = [0, 1 / 458259, 1 / 455150, 1 / 269909]
        else:
            weights = [0, 1 / 227021, 1 / 12176, 1 / 257733, 1 / 46275, 1 / 212, 1 / 408663, 1 / 97716, 1 / 133522]
        class_weights = torch.FloatTensor(weights)
        c_loss = nn.CrossEntropyLoss(reduction=self.reduction, weight=class_weights)
        loss = c_loss(y_pred, y_true)

        return loss
