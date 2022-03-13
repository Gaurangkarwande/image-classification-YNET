import torch
import torchvision
from torchvision import transforms
import time
import  numpy as np
import logging
import os
import gc
from matplotlib import pyplot as plt
import sys

from utils import EarlyStopping, LRScheduler
from models.YNet import YNet


def save_checkpoint(state, directory, file_name):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, file_name + '.pth')
    torch.save(state, checkpoint_file)

def train(dataloader, model, criterion, optimizer, device):
    loss_epoch = correct = total = 0
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        X, y = batch
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        _, pred_classes = pred.max(1)
        total += y.size(0)
        correct += float(pred_classes.eq(y).sum().item())
        optimizer.zero_grad()
    accuracy = correct/total
    avg_loss = loss_epoch/len(dataloader)
    return avg_loss, accuracy

def test(dataloader, model, criterion, device):
    loss_epoch = correct = total = 0
    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        X, y = batch
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        loss_epoch += loss.item()
        _, pred_classes = pred.max(1)
        total += y.size(0)
        correct += float(pred_classes.eq(y).sum().item())
    accuracy = correct/total
    avg_loss = loss_epoch/len(dataloader)
    return avg_loss, accuracy


def main(dataset='cifar10'):
    start = time.time()
    np.random.seed(345)
    torch.manual_seed(345)
    logging.basicConfig(filename=f'./logs/training_{dataset}.log', format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logging.info('\n************************************\n')
    print('\n************************************\n')
    data_dir = '/home/gaurangajitk/DL/data'
    data_root = os.path.join(data_dir, dataset)
    num_classes = 10 if dataset=='cifar10' else 100

    config = {
        "n_classes": num_classes,
        "batch_size": 128,
        "lr": 1e-3,
        "gradient_clip_val": 0.5,
        "num_epochs": 50,
        "cnn1_in": 3,
        "cnn2_in": 32,
        "cnn3_in": 64,
        "cnn3_out": 128,
        "linear_in": 1024,
        "dropout": 0.5,
        "kernel_size": 3
    }

    print(config)
    logging.info(config)
    print('Preparing Datasets')
    data_transforms = transforms.Compose([ transforms.ToTensor(),\
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                        download=True, transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=1)
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                        download=True, transform=data_transforms)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=True, num_workers=1)

    print('Creare Model')
    model = YNet(config)

    print('Setup criterion and optimizer')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    lr_scheduler = LRScheduler(optimizer)
    early_stopping = EarlyStopping()

    print('Check CUDA')
    train_on_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if(train_on_gpu):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')
    
    print('***** Training *****')
    logging.info('Started Training')

    model.to(device)

    best_valid_acc = 0
    train_history_loss = []
    train_history_acc = []
    val_history_loss = []
    val_history_acc = []

    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
        valid_loss, valid_acc = test(test_loader, model, criterion, device)
        time_for_epoch = time.time() - epoch_start

        print(f'Epoch {epoch}: Train Loss= {train_loss:.3f}, Train Acc= {train_acc:.3f} \t Valid Loss= {valid_loss:.3f}, Valid Acc= {valid_acc:.3f} \t Time Taken={time_for_epoch:.2f} s')
        logging.info(
            f'Epoch {epoch}: Train Loss= {train_loss:.3f}, Train Acc= {train_acc:.3f} \t Valid Loss= {valid_loss:.3f}, Valid Acc= {valid_acc:.3f} \t Time Taken={time_for_epoch:.2f} s')
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            checkpoint = {
                        'epoch': epoch, 
                        'model': model.state_dict(), 
                        'criterion': criterion.state_dict(), 
                        'optimizer': optimizer.state_dict(),
                        'best_acc': best_valid_acc
                        }
            save_checkpoint(checkpoint, directory='./model_checkpoint', file_name=f'best_checkpoint_{dataset}')
            logging.info(f'Checkpoint saved at Epoch {epoch}')

        lr_scheduler(valid_loss)
        early_stopping(valid_loss)
        #save losses for learning curves
        train_history_loss.append(train_loss)
        val_history_loss.append(valid_loss)
        train_history_acc.append(train_acc)
        val_history_acc.append(valid_acc)
        if early_stopping.early_stop:
            break
    del model; del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    logging.info(f'Final scheduler state {lr_scheduler.get_final_lr()}\n')

    # save curves
    plt.plot(range(len(train_history_loss)),train_history_loss, label="Training")
    plt.plot(range(len(val_history_loss)),val_history_loss, label="Validation")
    plt.legend()
    plt.title(f"Loss Curves: {dataset}")
    plt.savefig(f'curves/loss_curves_{dataset}.jpg', bbox_inches='tight', dpi=150)
    plt.close()

    plt.plot(range(len(train_history_acc)),train_history_acc, label="Training")
    plt.plot(range(len(val_history_acc)),val_history_acc, label="Validation")
    plt.legend()
    plt.title(f"Accuracy Curves: {dataset}")
    plt.savefig(f'curves/acc_curves_{dataset}.jpg', bbox_inches='tight', dpi=150)
    plt.close()

    print('***** Testing ********')


    PATH = f"/home/gaurangajitk/DL/cnn_YNet/model_checkpoint/best_checkpoint_{dataset}.pth"
    checkpoint = torch.load(PATH)
    model = YNet(config)
    model.to(device)
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    test_loss, test_acc = test(test_loader, model, criterion, device)
    print(f'Test Loss= {test_loss}, Test Acc= {test_acc}')
    logging.info(f'Test Loss= {test_loss}, Test Acc= {test_acc}')
    diff = time.time() - start
    logging.info(f'Total time taken= {str(diff)} s')
    print(f'Total time taken= {str(diff)} s')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = 'cifar10'
    main(dataset)