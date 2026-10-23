from collections.abc import Callable

import torch
from torch import nn

from deep_learning.config import DEVICE, LEARNING_RATE, MODEL_PATH, NUM_EPOCHS
from deep_learning.data.dataset import (
    test_data_loader,
    train_data_loader,
    validation_data_loader,
)
from deep_learning.engine.trainer import evaluate, train_one_epoch
from deep_learning.models.simple_cnn import SimpleCNN


def train_model(
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    best_loss: float | None = None

    for epoch in range(NUM_EPOCHS):
        epoch_loss, epoch_acc = train_one_epoch(
            model,
            data_loader=train_data_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        eval_loss, eval_acc = evaluate(
            model,
            data_loader=validation_data_loader,
            criterion=criterion,
            device=device,
        )

        if best_loss is None or eval_loss < best_loss:
            best_loss = eval_loss

            torch.save(model.state_dict(), MODEL_PATH)

        print(f" Epoch {epoch + 1} ".center(60, "="))
        print(f"Training Loss / Accuracy: {epoch_loss:.4f} / {epoch_acc:.2%}")
        print(f"Validation Loss / Accuracy: {eval_loss:.4f} / {eval_acc:.2%}")


def test_model(
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
):
    test_loss, test_acc = evaluate(
        model, data_loader=test_data_loader, criterion=criterion, device=device
    )

    print(" Final Validation ".center(60, "="))
    print(f"Training Loss / Accuracy: {test_loss:.4f} / {test_acc:.2%}")


if __name__ == "__main__":
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, criterion=criterion, device=DEVICE)

    trained_model = SimpleCNN().to(DEVICE)
    trained_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    test_model(trained_model, criterion=criterion, device=DEVICE)
