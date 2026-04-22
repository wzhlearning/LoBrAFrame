import torch
import numpy as np
import os
from LoBrAFrame import LoBrAMamba
from dataloader import Dataset, load_all_data, LOSO_train_test_set
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, precision_score, recall_score, f1_score


# Select dataset through task_id
task = ['UFFT', 'MA', 'WG']
task_id = 0
print(task[task_id])

# Set dataset path
UFFT_data_path = 'UFFT_data'
MA_data_path = 'MA_fNIRS_data'
WG_data_path = 'WG_fNIRS_data'

if task_id == 0:
    # UFFT
    num_class = 3  # number of classes: RHT, LHT, and FT
    EPOCH = 30  # number of training epoch
    all_sub = 30  # number of subjects
    batch_size = 512
    lr = 1e-3
    data_path = UFFT_data_path

elif task_id == 1:
    # MA
    num_class = 2  # number of classes: MA and BL
    EPOCH = 30  # number of training epoch
    all_sub = 29  # number of subjects
    batch_size = 512
    lr = 5e-4
    data_path = MA_data_path

elif task_id == 2:
    # WG
    num_class = 2  # number of classes: WG and BL
    EPOCH = 30  # number of training epoch
    all_sub = 26  # number of subjects
    batch_size = 512
    lr = 5e-4
    data_path = WG_data_path


root_path = os.path.join('save',  task[task_id], 'LOSO')
while (os.path.exists(root_path) is True):
    print('path is exist')
os.makedirs(root_path)

all_data, all_label = load_all_data(data_path, task_id)
for n_sub in range(all_sub):
    path = os.path.join(root_path, str(n_sub+1))
    while (os.path.exists(path) is True):
        print('sub path is exist')
    os.makedirs(path)

    # load dataset
    X_train, y_train, X_test, y_test = LOSO_train_test_set(all_data, all_label, n_sub, task_id)
    train_set = Dataset(X_train, y_train, transform=False)
    test_set = Dataset(X_test, y_test, transform=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=X_test.shape[0], shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # LoBrAMamba
    if task_id == 0:
        net = LoBrAMamba(n_classes=num_class, signal_channel=20, signal_length=40, HL_layers=2).to(device)
    elif task_id == 1:
        net = LoBrAMamba(n_classes=num_class, signal_channel=36, signal_length=30, HL_layers=2).to(device)
    elif task_id == 2:
        net = LoBrAMamba(n_classes=num_class, signal_channel=36, signal_length=30, HL_layers=2).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    lrStep = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    metrics = open(path + '/metrics.txt', 'w')
    # -------------------------------------------------------------------------------------------------------------------- #
    # model training
    for epoch in range(EPOCH):
        net.train()
        train_running_acc = 0
        total = 0
        loss_steps = []
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_steps.append(loss.item())
            total += labels.shape[0]
            pred = outputs.argmax(dim=1, keepdim=True)
            train_running_acc += pred.eq(labels.view_as(pred)).sum().item()

        train_running_loss = float(np.mean(loss_steps))
        train_running_acc = 100 * train_running_acc / total
        print('[%d, %d] Train loss: %0.5f' % (n_sub+1, epoch, train_running_loss))
        print('[%d, %d] Train acc: %0.3f%%' % (n_sub+1, epoch, train_running_acc))

        # -------------------------------------------------------------------------------------------------------------------- #
        # model evaluation
        net.eval()
        test_running_acc = 0
        total = 0
        loss_steps = []
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels.long())

                loss_steps.append(loss.item())
                total += labels.shape[0]
                pred = outputs.argmax(dim=1, keepdim=True)
                test_running_acc += pred.eq(labels.view_as(pred)).sum().item()

            test_running_acc = 100 * test_running_acc / total
            test_running_loss = float(np.mean(loss_steps))
            print('     [%d, %d] Test loss: %0.5f' % (n_sub+1, epoch, test_running_loss))
            print('     [%d, %d] Test acc: %0.3f%%' % (n_sub+1, epoch, test_running_acc))

            y_label = labels.cpu()
            y_pred = pred.cpu()
            acc = accuracy_score(y_label, y_pred)
            if task_id == 0:
                # macro mode for UFFT
                precision = precision_score(y_label, y_pred, average='macro')
                recall = recall_score(y_label, y_pred, average='macro')
                f1 = f1_score(y_label, y_pred, average='macro')
            elif task_id == 1 or 2:
                # MA and WG
                precision = precision_score(y_label, y_pred)
                recall = recall_score(y_label, y_pred)
                f1 = f1_score(y_label, y_pred)
            kappa_value = cohen_kappa_score(y_label, y_pred)
            confusion = confusion_matrix(y_label, y_pred)
            metrics.write("acc=%.4f, pre=%.4f, rec=%.4f, f1=%.4f, kap=%.4f" % (acc*100, precision*100, recall*100, f1, kappa_value))
            metrics.write('\n')
            metrics.flush()

        # save model weight
        torch.save(net.state_dict(), os.path.join(path, 'model.pt'))

        lrStep.step()

