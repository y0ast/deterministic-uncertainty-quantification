import argparse
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from utils.datasets import all_datasets
from utils.cnn_duq import SoftmaxModel as CNN
from torchvision.models import resnet18


class ResNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.resnet = resnet18(pretrained=False, num_classes=num_classes)

        # Adapted resnet from:
        # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        x = F.log_softmax(x, dim=1)

        return x


def train(model, train_loader, optimizer, epoch, loss_fn):
    model.train()

    total_loss = []

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        prediction = model(data)
        loss = loss_fn(prediction, target)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = torch.tensor(total_loss).mean()
    print(f"Epoch: {epoch}:")
    print(f"Train Set: Average Loss: {avg_loss:.2f}")


def test(models, test_loader, loss_fn):
    models.eval()

    loss = 0
    correct = 0

    for data, target in test_loader:
        with torch.no_grad():
            data = data.cuda()
            target = target.cuda()

            losses = torch.empty(len(models), data.shape[0])
            predictions = []
            for i, model in enumerate(models):
                predictions.append(model(data))
                losses[i, :] = loss_fn(predictions[i], target, reduction="sum")

            predictions = torch.stack(predictions)

            loss += torch.mean(losses)
            avg_prediction = predictions.exp().mean(0)

            # get the index of the max log-probability
            class_prediction = avg_prediction.max(1)[1]
            correct += (
                class_prediction.eq(target.view_as(class_prediction)).sum().item()
            )

    loss /= len(test_loader.dataset)

    percentage_correct = 100.0 * correct / len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(test_loader.dataset), percentage_correct
        )
    )

    return loss, percentage_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=75, help="number of epochs to train (default: 75)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.05, help="learning rate (default: 0.05)"
    )
    parser.add_argument(
        "--ensemble", type=int, default=5, help="Ensemble size (default: 5)"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["FashionMNIST", "CIFAR10"],
        help="Select a dataset",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    loss_fn = F.nll_loss

    ds = all_datasets[args.dataset]()
    input_size, num_classes, train_dataset, test_dataset = ds

    kwargs = {"num_workers": 4, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=5000, shuffle=False, **kwargs
    )

    if args.dataset == "FashionMNIST":
        milestones = [10, 20]
        ensemble = [CNN(input_size, num_classes).cuda() for _ in range(args.ensemble)]
    else:
        # CIFAR-10
        milestones = [25, 50]
        ensemble = [
            ResNet(input_size, num_classes).cuda() for _ in range(args.ensemble)
        ]

    ensemble = torch.nn.ModuleList(ensemble)

    optimizers = []
    schedulers = []

    for model in ensemble:
        # Need different optimisers to apply weight decay and momentum properly
        # when only optimising one element of the ensemble
        optimizers.append(
            torch.optim.SGD(
                model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
            )
        )

        schedulers.append(
            torch.optim.lr_scheduler.MultiStepLR(
                optimizers[-1], milestones=milestones, gamma=0.1
            )
        )

    for epoch in range(1, args.epochs + 1):
        for i, model in enumerate(ensemble):
            train(model, train_loader, optimizers[i], epoch, loss_fn)
            schedulers[i].step()

        test(ensemble, test_loader, loss_fn)

    pathlib.Path("saved_models").mkdir(exist_ok=True)
    path = f"saved_models/{args.dataset}_{len(ensemble)}"
    torch.save(ensemble.state_dict(), path + "_ensemble.pt")


if __name__ == "__main__":
    main()
