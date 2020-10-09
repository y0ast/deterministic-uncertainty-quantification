import argparse
import json
import pathlib
import random

import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard.writer import SummaryWriter

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average, Loss
from ignite.contrib.handlers import ProgressBar

from utils.resnet_duq import ResNet_DUQ
from utils.datasets import all_datasets
from utils.evaluate_ood import get_cifar_svhn_ood, get_auroc_classification


def main(
    batch_size,
    epochs,
    length_scale,
    centroid_size,
    model_output_size,
    learning_rate,
    l_gradient_penalty,
    gamma,
    weight_decay,
    final_model,
):
    name = f"DUQ_{length_scale}__{l_gradient_penalty}_{gamma}_{centroid_size}"
    writer = SummaryWriter(comment=name)

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

    model = ResNet_DUQ(
        input_size, num_classes, centroid_size, model_output_size, length_scale, gamma
    )
    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 50, 75], gamma=0.2
    )

    def bce_loss_fn(y_pred, y):
        bce = F.binary_cross_entropy(y_pred, y, reduction="sum").div(
            num_classes * y_pred.shape[0]
        )
        return bce

    def output_transform_bce(output):
        y_pred, y, x = output

        y = F.one_hot(y, num_classes).float()

        return y_pred, y

    def output_transform_acc(output):
        y_pred, y, x = output

        return y_pred, y

    def output_transform_gp(output):
        y_pred, y, x = output

        return x, y_pred

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

        if l_gradient_penalty > 0:
            x.requires_grad_(True)

        z, y_pred = model(x)
        y = F.one_hot(y, num_classes).float()

        loss = bce_loss_fn(y_pred, y)

        if l_gradient_penalty > 0:
            loss += l_gradient_penalty * calc_gradient_penalty(x, y_pred)

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

        z, y_pred = model(x)

        return y_pred, y, x

    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Average()
    metric.attach(trainer, "loss")

    metric = Accuracy(output_transform=output_transform_acc)
    metric.attach(evaluator, "accuracy")

    metric = Loss(F.binary_cross_entropy, output_transform=output_transform_bce)
    metric.attach(evaluator, "bce")

    metric = Loss(calc_gradient_penalty, output_transform=output_transform_gp)
    metric.attach(evaluator, "gradient_penalty")

    kwargs = {"num_workers": 4, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1000, shuffle=False, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False, **kwargs
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        metrics = trainer.state.metrics
        loss = metrics["loss"]

        print(f"Train - Epoch: {trainer.state.epoch} Loss: {loss:.2f} ")

        writer.add_scalar("Loss/train", loss, trainer.state.epoch)

        if trainer.state.epoch % 5 == 0 or trainer.state.epoch > 65:
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

        print(f"Centroid norm: {torch.norm(model.m / model.N, dim=0)}")

        scheduler.step()

        if trainer.state.epoch > 65:
            torch.save(
                model.state_dict(), f"saved_models/{name}_{trainer.state.epoch}.pt"
            )

    pbar = ProgressBar(dynamic_ncols=True)
    pbar.attach(trainer)

    trainer.run(train_loader, max_epochs=epochs)

    evaluator.run(test_loader)
    acc = evaluator.state.metrics["accuracy"]

    print(f"Test - Accuracy {acc:.4f}")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", type=int, default=75, help="Number of epochs to train (default: 75)"
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
        default=512,
        help="Size to use for centroids (default: 512)",
    )

    parser.add_argument(
        "--model_output_size",
        type=int,
        default=512,
        help="Size to use for model output (default: 512)",
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
        default=0.5,
        help="Weight for gradient penalty (default: 0.5)",
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

    pathlib.Path("saved_models").mkdir(exist_ok=True)

    main(**kwargs)
