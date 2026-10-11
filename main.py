import argparse

import torch
import wandb
from pathlib import Path
from torch import nn

from model import CNN, build_resnet
from utils import classification_metrics, load_data, seed_everything, write_artifacts


def evaluate(model, loader, criterion, device, num_classes=10):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(1)

            total_loss += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            for label, prediction in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
                confusion[label, prediction] += 1

    return total_loss / total, correct / total, confusion.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="nueral networks")
    parser.add_argument("-num_epoches", dest="num_epoches", type=int, default=40, help="num of epoches")
    parser.add_argument("-fc_hidden1", dest="fc_hidden1", type=int, default=100, help="dim of hidden neurons")
    parser.add_argument("-fc_hidden2", dest="fc_hidden2", type=int, default=100, help="dim of hidden neurons")
    parser.add_argument("-learning_rate", dest="learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("-decay", dest="decay", type=float, default=0.5, help="learning rate decay factor every 10 epochs")
    parser.add_argument("-batch_size", dest="batch_size", type=int, default=100, help="batch size")
    parser.add_argument("-dropout", dest="dropout", type=float, default=0.4, help="dropout prob")
    parser.add_argument("-rotation", dest="rotation", type=int, default=10, help="image rotation")
    parser.add_argument("-activation", dest="activation", type=str, default='relu',
                        choices=['relu', 'leaky_relu', 'tanh', 'sigmoid'], help="activation function")
    parser.add_argument("-channel_out1", dest="channel_out1", type=int, default=64, help="number of channels")
    parser.add_argument("-channel_out2", dest="channel_out2", type=int, default=64, help="number of channels")
    parser.add_argument("-k_size", dest="k_size", type=int, default=4, help="size of filter")
    parser.add_argument("-pooling_size", dest="pooling_size", type=int, default=2, help="size for max pooling")
    parser.add_argument("-stride", dest="stride", type=int, default=1, help="stride for filter")
    parser.add_argument("-max_stride", dest="max_stride", type=int, default=2, help="stride for max pooling")
    parser.add_argument("-wandb_mode", choices=['online', 'offline'], default='online', help="W&B logging mode")
    parser.add_argument("-bonus", type=int, choices=[18, 34, 50], nargs='?', const=18,
                        help="train ResNet-18, 34, or 50 on CIFAR-100 (default: 18)")
    args = parser.parse_args()
    if args.num_epoches < 1:
        parser.error("-num_epoches must be at least 1")

    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folder = Path(__file__).parent
    seed_everything(seed)
    dataset = 'cifar100' if args.bonus else 'cifar10'
    train_loader, val_loader, test_loader, classes = load_data(
        folder / 'data', args.batch_size, workers=4, seed=seed,
        download=True, rotation=args.rotation, dataset=dataset
    )
    num_classes = len(classes)
    if args.bonus:
        model = build_resnet(args.bonus, num_classes).to(device)
    else:
        model = CNN(dropout=args.dropout, fc_hidden1=args.fc_hidden1,
                    fc_hidden2=args.fc_hidden2, activation=args.activation,
                    channel_out1=args.channel_out1, channel_out2=args.channel_out2,
                    k_size=args.k_size, pooling_size=args.pooling_size,
                    stride=args.stride, max_stride=args.max_stride).to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0005)
    if args.bonus:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=0.0005,
            nesterov=True
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[60, 120, 160], gamma=args.decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.0005
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=args.decay
        )
    run = wandb.init(project=f'cnn-assignment-{dataset}', mode=args.wandb_mode,
                     dir=str(folder), config=vars(args))
    history = []

    for epoch in range(args.num_epoches):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_accuracy = correct / total
        val_loss, val_accuracy, _ = evaluate(model, val_loader, criterion, device, num_classes)

        row = {'epoch': epoch + 1, 'learning_rate': optimizer.param_groups[0]['lr'],
               'train_loss': train_loss, 'train_accuracy': train_accuracy,
               'val_loss': val_loss, 'val_accuracy': val_accuracy}
        history.append(row)
        run.log(row, step=epoch + 1)
        print(f'Epoch {epoch + 1}: loss={train_loss:.4f}, '
              f'train={train_accuracy:.2%}, val={val_accuracy:.2%}')
        scheduler.step()

    for metric in ('loss', 'accuracy'):
        chart = wandb.plot.line_series(
            xs=[row['epoch'] for row in history],
            ys=[[row[f'{split}_{metric}'] for row in history] for split in ('train', 'val')],
            keys=['train', 'validation'], title=f'Training and validation {metric}', xname='Epoch'
        )
        run.log({f'{metric}_curves': chart})

    test_loss, test_accuracy, confusion = evaluate(model, test_loader, criterion, device, num_classes)
    metrics = classification_metrics(confusion, classes)
    print(f'Test accuracy: {test_accuracy:.2%}')
    print(f"Precision: {metrics['macro_precision']:.4f}")
    print(f"Recall: {metrics['macro_recall']:.4f}")
    print(f"F1: {metrics['macro_f1']:.4f}")
    run.summary.update({'test_loss': test_loss, 'test_accuracy': test_accuracy,
                        'test_precision': metrics['macro_precision'],
                        'test_recall': metrics['macro_recall'], 'test_f1': metrics['macro_f1']})
    results = folder / 'output' / run.id
    metrics['test_loss'] = test_loss
    write_artifacts(results, history, metrics)
    print(f'Results saved to {results}')
    run.finish()
