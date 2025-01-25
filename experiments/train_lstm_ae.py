# experiments/train_lstm_ae.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import sys
from typing import Dict
import time

# Make sure we can import from top-level
sys.path.append("..")

from model.rae import LSTMAutoencoder
from data_factory.data_loader import get_loader_segment
from utils import compute_scores, adjust_learning_rate, detection_adjustment


def train_and_eval_lstm_autoencoder(
    hidden_dim1: int,
    hidden_dim2: int,
    lr: float = 1e-3,
    num_epochs: int = 10,
    batch_size: int = 256,
    anormly_ratio: float = 0.5,
    win_size: int = 100,
    input_c: int = 38,
    output_c: int = 38,
    dataset_name: str = "SMD",
    data_path: str = "./dataset/SMD",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """
    Train and evaluate the LSTMAutoencoder on the SMD dataset with given hyperparameters.
    We'll do a straightforward reconstruction-based anomaly detection:
    - compute reconstruction error
    - threshold
    - evaluate
    """

    # 1) Data
    train_loader = get_loader_segment(
        data_path,
        batch_size,
        win_size=win_size,
        step=1,
        mode="train",
        dataset=dataset_name,
    )
    test_loader = get_loader_segment(
        data_path,
        batch_size,
        win_size=win_size,
        step=1,
        mode="test",
        dataset=dataset_name,
    )
    thre_loader = get_loader_segment(
        data_path,
        batch_size,
        win_size=win_size,
        step=1,
        mode="thre",
        dataset=dataset_name,
    )

    # 2) Model
    model = LSTMAutoencoder(
        input_dim=input_c,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        output_dim=output_c,
        dropout=0.0,
        layer_norm_flag=False,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Optional: a simple validation function for LSTM using the test loader
    def vali_lstm_autoenc(model, loader, device):
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for input_data, _ in loader:
                input_data = input_data.float().to(device)
                output_data = model(input_data)
                loss = criterion(output_data, input_data)
                total_loss += loss.item()
                count += 1
        return total_loss / max(count, 1)

    # 3) Training with logs
    print("==== Starting LSTMAutoencoder Training ====")
    time_now = time.time()
    train_steps = len(train_loader)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        iter_count = 0
        loss_list = []
        epoch_start_time = time.time()

        for i, (input_data, _) in enumerate(train_loader):
            iter_count += 1
            input_data = input_data.float().to(device)

            optimizer.zero_grad()
            output_data = model(input_data)
            loss_mse = criterion(output_data, input_data)
            loss_mse.backward()
            optimizer.step()

            loss_list.append(loss_mse.item())

            # Print speed & left time every 100 steps
            if (i + 1) % 100 == 0:
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((num_epochs - epoch) * train_steps - i)
                print(
                    f"\tIter: {i + 1}/{train_steps}, speed: {speed:.4f}s/iter; "
                    f"left time: {left_time:.4f}s"
                )
                iter_count = 0
                time_now = time.time()

        epoch_time_used = time.time() - epoch_start_time
        train_loss_epoch = np.mean(loss_list)
        train_losses.append(train_loss_epoch)
        # "Validate" with test loader
        vali_loss = vali_lstm_autoenc(model, test_loader, device)
        val_losses.append(vali_loss)
        print(
            f"Epoch: {epoch + 1}, Time: {epoch_time_used:.2f}s, "
            f"Train Loss: {train_loss_epoch:.6f}, Vali Loss: {vali_loss:.6f}"
        )

    # 4) Evaluate anomaly detection
    model.eval()
    local_criterion = nn.MSELoss(reduction="none")

    # (a) Train set reconstruction errors
    train_errs = []
    for input_data, _ in train_loader:
        input_data = input_data.float().to(device)
        output_data = model(input_data)
        errs = torch.mean(
            local_criterion(input_data, output_data), dim=-1
        )  # shape: (B, win_size)
        train_errs.append(errs.detach().cpu().numpy())
    train_errs = np.concatenate(train_errs, axis=0).reshape(-1)

    # (b) threshold from train + portion of test
    thr_errs = []
    for input_data, _ in thre_loader:
        input_data = input_data.float().to(device)
        output_data = model(input_data)
        errs = torch.mean(local_criterion(input_data, output_data), dim=-1)
        thr_errs.append(errs.detach().cpu().numpy())
    thr_errs = np.concatenate(thr_errs, axis=0).reshape(-1)

    combined_errs = np.concatenate([train_errs, thr_errs], axis=0)
    thresh = np.percentile(combined_errs, 100 - anormly_ratio)

    # (c) final test predictions
    preds_list = []
    test_labels_list = []
    for input_data, labels in thre_loader:
        input_data = input_data.float().to(device)
        out_data = model(input_data)
        errs = torch.mean(local_criterion(input_data, out_data), dim=-1)
        preds_list.append(errs.detach().cpu().numpy())
        test_labels_list.append(labels.numpy())

    preds = np.concatenate(preds_list, axis=0).reshape(-1)
    gt = np.concatenate(test_labels_list, axis=0).reshape(-1)

    y_hat = (preds > thresh).astype(int)
    y_hat = detection_adjustment(y_hat, gt)
    scores = compute_scores(gt, y_hat)

    print("==== LSTMAutoencoder Final Results ====")
    print(f"Threshold: {thresh:.6f}")
    print(
        f"Accuracy: {scores['accuracy']:.4f}, Precision: {scores['precision']:.4f}, "
        f"Recall: {scores['recall']:.4f}, F1: {scores['f1_score']:.4f}"
    )

    results = {
        "train_error_mean": float(train_errs.mean()),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "threshold": float(thresh),
        "scores": scores,
    }
    return results, model
