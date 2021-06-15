import argparse
import json
import pathlib
import random

import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard.writer import SummaryWriter

from torchvision.models import resnet18

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average, Loss
from ignite.contrib.handlers import ProgressBar

from utils.wide_resnet import WideResNet
from utils.resnet_duq import ResNet_DUQ
from utils.datasets import all_datasets
from utils.evaluate_ood import get_cifar_svhn_ood, get_auroc_classification


def main(
    architecture,
    batch_size,
    length_scale,
    centroid_size,
    learning_rate,
    l_gradient_penalty,
    gamma,
    weight_decay,
    final_model,
    output_dir,
):
    writer = SummaryWriter(log_dir=f"runs/{output_dir}")

    ds = all_datasets["CIFAR10"]()
    input_size, num_classes, dataset, test_dataset = ds

    # Split up training set
    idx = list(range(len(dataset)))
    random.shuffle(idx)

    if final_model:
        train_dataset = dataset
        val_dataset = test_dataset
    else:
        val_size = int(len(dataset) * 0.8)
        train_dataset = torch.utils.data.Subset(dataset, idx[:val_size])
        val_dataset = torch.utils.data.Subset(dataset, idx[val_size:])

        val_dataset.transform = (
            test_dataset.transform
        )  # Test time preprocessing for validation

    if architecture == "WRN":
        model_output_size = 640
        epochs = 200
        milestones = [60, 120, 160]
        feature_extractor = WideResNet()
    elif architecture == "ResNet18":
        model_output_size = 512
        epochs = 100
        milestones = [25, 50, 75]
        feature_extractor = resnet18()

        # Adapted resnet from:
        # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        feature_extractor.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        feature_extractor.maxpool = torch.nn.Identity()
        feature_extractor.fc = torch.nn.Identity()

    if centroid_size is None:
        centroid_size = model_output_size

    model = ResNet_DUQ(
        feature_extractor,
        num_classes,
        centroid_size,
        model_output_size,
        length_scale,
        gamma,
    )
    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.2
    )

    def calc_gradients_input(x, y_pred):
        gradients = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
        )[0]

        gradients = gradients.flatten(start_dim=1)

        return gradients

    def calc_gradient_penalty(x, y_pred):
        gradients = calc_gradients_input(x, y_pred)

        # L2 norm
        grad_norm = gradients.norm(2, dim=1)

        # Two sided penalty
        gradient_penalty = ((grad_norm - 1) ** 2).mean()

        return gradient_penalty

    def step(engine, batch):
        model.train()

        optimizer.zero_grad()

        x, y = batch
        x, y = x.cuda(), y.cuda()

        x.requires_grad_(True)

        y_pred = model(x)

        y = F.one_hot(y, num_classes).float()

        loss = F.binary_cross_entropy(y_pred, y, reduction="mean")

        if l_gradient_penalty > 0:
            gp = calc_gradient_penalty(x, y_pred)
            loss += l_gradient_penalty * gp

        loss.backward()
        optimizer.step()

        x.requires_grad_(False)

        with torch.no_grad():
            model.eval()
            model.update_embeddings(x, y)

        return loss.item()

    def eval_step(engine, batch):
        model.eval()

        x, y = batch
        x, y = x.cuda(), y.cuda()

        x.requires_grad_(True)

        y_pred = model(x)

        return {"x": x, "y": y, "y_pred": y_pred}

    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Average()
    metric.attach(trainer, "loss")

    metric = Accuracy(output_transform=lambda out: (out["y_pred"], out["y"]))
    metric.attach(evaluator, "accuracy")

    def bce_output_transform(out):
        return (out["y_pred"], F.one_hot(out["y"], num_classes).float())

    metric = Loss(F.binary_cross_entropy, output_transform=bce_output_transform)
    metric.attach(evaluator, "bce")

    metric = Loss(
        calc_gradient_penalty, output_transform=lambda out: (out["x"], out["y_pred"])
    )
    metric.attach(evaluator, "gradient_penalty")

    pbar = ProgressBar(dynamic_ncols=True)
    pbar.attach(trainer)

    kwargs = {"num_workers": 4, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        metrics = trainer.state.metrics
        loss = metrics["loss"]

        print(f"Train - Epoch: {trainer.state.epoch} Loss: {loss:.2f}")

        writer.add_scalar("Loss/train", loss, trainer.state.epoch)

        if trainer.state.epoch > (epochs - 5):
            accuracy, auroc = get_cifar_svhn_ood(model)
            print(f"Test Accuracy: {accuracy}, AUROC: {auroc}")
            writer.add_scalar("OoD/test_accuracy", accuracy, trainer.state.epoch)
            writer.add_scalar("OoD/roc_auc", auroc, trainer.state.epoch)

            accuracy, auroc = get_auroc_classification(val_dataset, model)
            print(f"AUROC - uncertainty: {auroc}")
            writer.add_scalar("OoD/val_accuracy", accuracy, trainer.state.epoch)
            writer.add_scalar("OoD/roc_auc_classification", auroc, trainer.state.epoch)

        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        acc = metrics["accuracy"]
        bce = metrics["bce"]
        GP = metrics["gradient_penalty"]
        loss = bce + l_gradient_penalty * GP

        print(
            (
                f"Valid - Epoch: {trainer.state.epoch} "
                f"Acc: {acc:.4f} "
                f"Loss: {loss:.2f} "
                f"BCE: {bce:.2f} "
                f"GP: {GP:.2f} "
            )
        )

        writer.add_scalar("Loss/valid", loss, trainer.state.epoch)
        writer.add_scalar("BCE/valid", bce, trainer.state.epoch)
        writer.add_scalar("GP/valid", GP, trainer.state.epoch)
        writer.add_scalar("Accuracy/valid", acc, trainer.state.epoch)

        scheduler.step()

    trainer.run(train_loader, max_epochs=epochs)
    evaluator.run(test_loader)
    acc = evaluator.state.metrics["accuracy"]

    print(f"Test - Accuracy {acc:.4f}")

    torch.save(model.state_dict(), f"runs/{output_dir}/model.pt")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--architecture",
        default="ResNet18",
        choices=["ResNet18", "WRN"],
        help="Pick an architecture (default: ResNet18)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size to use for training (default: 128)",
    )

    parser.add_argument(
        "--centroid_size",
        type=int,
        default=None,
        help="Size to use for centroids (default: same as model output)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.05,
        help="Learning rate (default: 0.05)",
    )

    parser.add_argument(
        "--l_gradient_penalty",
        type=float,
        default=0.75,
        help="Weight for gradient penalty (default: 0.75)",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help="Decay factor for exponential average (default: 0.999)",
    )

    parser.add_argument(
        "--length_scale",
        type=float,
        default=0.1,
        help="Length scale of RBF kernel (default: 0.1)",
    )

    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="Weight decay (default: 5e-4)"
    )

    parser.add_argument(
        "--output_dir", type=str, default="results", help="set output folder"
    )

    # Below setting cannot be used for model selection,
    # because the validation set equals the test set.
    parser.add_argument(
        "--final_model",
        action="store_true",
        default=False,
        help="Use entire training set for final model",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    print("input args:\n", json.dumps(kwargs, indent=4, separators=(",", ":")))

    pathlib.Path("runs/" + args.output_dir).mkdir(exist_ok=True)

    main(**kwargs)
