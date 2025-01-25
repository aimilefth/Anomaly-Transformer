# experiments/train_at.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import sys
from typing import Dict
import time

# Make sure we can import from top-level
sys.path.append("..")

from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
from utils import (
    my_kl_loss,
    adjust_learning_rate,
    compute_scores,
    detection_adjustment,
)


###############################################################################
# Explanation of AnomalyTransformer Training/Evaluation
###############################################################################
"""
**AnomalyTransformer Explanation**:

During training, the AnomalyTransformer outputs:

1) `output`: the reconstructed time-series (shape: [B, L, D]).  
2) `series[u]`: the "series attention map" from the u-th encoder layer.  
3) `prior[u]`: the "prior attention map" from the u-th encoder layer.

The model attempts to learn an association-discrepancy mechanism via KL divergence:
 - We interpret each attention map (series[u] or prior[u]) as a distribution over time steps.
 - We compute KL(p || q) in both directions on these distributions, then average.

**normalized_prior**:
  We take `prior[u]` and normalize it along the time dimension so that each row sums to 1.  
  `normalized_prior = prior[u] / sum_of_prior[u]` (with a small epsilon if needed).  

This results in a valid "distribution" that we can compare with the `series[u]` distribution using `KL divergence`.

**series_loss**:
  Summation of `KL(series[u], normalized_prior.detach()) + KL(normalized_prior.detach(), series[u])`
  across each layer `u`.  
  This measures how different the "series attention" distribution is from the "prior attention" distribution.

**prior_loss**:
  Summation of `KL(normalized_prior, series[u].detach()) + KL(series[u].detach(), normalized_prior)`.
  Conceptually similar, just reversing roles in the training objective for the min-max portion.

**Min-Max Training**:
  - We define two losses:
    L1 = reconstruction_loss - k * series_loss
    L2 = reconstruction_loss + k * prior_loss
  - We first backprop with L1, then also backprop with L2, effectively pushing the model to reduce the discrepancy from one side and increase it from the other side (the "association-discrepancy" concept).

**Temperature**:
  We multiply KL terms by a `temperature` (e.g. 50) during *inference*, not training, to scale the effect of the attention-based distribution. This follows the original AnomalyTransformer code.  
  It's basically a hyperparameter that can sharpen or flatten the distribution effect in the anomaly scoring.  

In **evaluation**:
 - We compute a reconstruction MSE along each time-step.  
 - We compute the "energy" (attens_energy) by weighting MSE with a negative association-discrepancy measure ( - series_loss - prior_loss ).  
 - We then apply `softmax` to this negative synergy to get a "metric" distribution, and multiply by the MSE.  
 - We gather energies from both train set and test set, pick a percentile threshold, and classify anomalies by `energy > threshold`.
"""


def train_and_eval_anomaly_transformer(
    d_model: int,
    n_heads: int,
    e_layers: int,
    d_ff: int,
    lr: float = 1e-4,
    k: float = 3.0,
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
    Train and evaluate the AnomalyTransformer on the SMD dataset with given hyperparameters.
    Now includes epoch-based logging similar to solver.py.
    """

    # -------------------------
    # 1. Prepare data loaders
    # -------------------------
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

    # -------------------------
    # 2. Create model and util
    # -------------------------
    model = AnomalyTransformer(
        win_size=win_size,
        enc_in=input_c,
        c_out=output_c,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_ff=d_ff,
        dropout=0.0,
        activation="gelu",
        output_attention=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    def vali_anomaly_transformer(model, loader, device, k, criterion):
        """
        A simple 'validation' that mimics solver.py's vali approach,
        but uses the test_loader to compute a rough measure of loss_1 and loss_2.
        """
        model.eval()
        loss_1_list = []
        loss_2_list = []
        with torch.no_grad():
            for input_data, _ in loader:
                input_data = input_data.float().to(device)
                output, series, prior, _ = model(input_data)

                # Compute series_loss and prior_loss (association discrepancy)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    # normalize prior
                    norm_prior = prior[u] / torch.unsqueeze(
                        torch.sum(prior[u], dim=-1), dim=-1
                    ).repeat(1, 1, 1, win_size)
                    series_loss += torch.mean(
                        my_kl_loss(series[u], norm_prior.detach())
                    )
                    series_loss += torch.mean(
                        my_kl_loss(norm_prior.detach(), series[u])
                    )

                    prior_loss += torch.mean(my_kl_loss(norm_prior, series[u].detach()))
                    prior_loss += torch.mean(my_kl_loss(series[u].detach(), norm_prior))

                series_loss /= len(prior)
                prior_loss /= len(prior)

                rec_loss = criterion(output, input_data)
                # same structure as solver vali
                loss1 = rec_loss - k * series_loss
                loss2 = rec_loss + k * prior_loss

                loss_1_list.append(loss1.item())
                loss_2_list.append(loss2.item())

        return np.mean(loss_1_list), np.mean(loss_2_list)

    # -------------------------
    # 3. Training Loop
    # -------------------------
    print("==== Starting AnomalyTransformer Training ====")
    time_now = time.time()
    train_steps = len(train_loader)
    train_losses = []
    val1_losses = []
    val2_losses = []
    for epoch in range(num_epochs):
        model.train()
        iter_count = 0
        loss1_list = []

        epoch_start_time = time.time()
        for i, (input_data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            iter_count += 1

            input_data = input_data.float().to(device)
            output, series, prior, _ = model(input_data)

            # Compute series_loss, prior_loss
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                # normalize prior along time dimension
                normalized_prior = prior[u] / torch.unsqueeze(
                    torch.sum(prior[u], dim=-1), dim=-1
                ).repeat(1, 1, 1, win_size)
                # KL in both directions
                series_loss += torch.mean(
                    my_kl_loss(series[u], normalized_prior.detach())
                )
                series_loss += torch.mean(
                    my_kl_loss(normalized_prior.detach(), series[u])
                )

                prior_loss += torch.mean(
                    my_kl_loss(normalized_prior, series[u].detach())
                )
                prior_loss += torch.mean(
                    my_kl_loss(series[u].detach(), normalized_prior)
                )

            series_loss /= len(prior)
            prior_loss /= len(prior)

            rec_loss = criterion(output, input_data)
            # Minimax
            loss1 = rec_loss - k * series_loss
            loss2 = rec_loss + k * prior_loss

            # store for average
            loss1_list.append(loss1.item())

            # Print speed & left time every 100 iters
            if (i + 1) % 100 == 0:
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((num_epochs - epoch) * train_steps - i)
                print(
                    f"\tIter: {i + 1}/{train_steps}, speed: {speed:.4f}s/iter; "
                    f"left time: {left_time:.4f}s"
                )
                iter_count = 0
                time_now = time.time()

            loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer.step()

        epoch_time_used = time.time() - epoch_start_time
        train_loss_epoch = np.mean(loss1_list)
        train_losses.append(train_loss_epoch)
        # 'Validate' with test_loader
        vali_loss1, vali_loss2 = vali_anomaly_transformer(
            model, test_loader, device, k, criterion
        )
        val1_losses.append(vali_loss1)
        val2_losses.append(vali_loss2)
        print(
            f"Epoch: {epoch + 1}, Time: {epoch_time_used:.2f}s, "
            f"Train Loss: {train_loss_epoch:.6f}, "
            f"Vali Loss1: {vali_loss1:.6f}, Vali Loss2: {vali_loss2:.6f}"
        )

        # You could optionally do LR adjust or early stopping here
        adjust_learning_rate(optimizer, epoch + 1, lr)

    # -------------------------
    # Final Evaluation
    # -------------------------
    model.eval()
    temperature = 50
    local_criterion = nn.MSELoss(reduction="none")

    # (a) gather train energy
    train_energy_list = []
    for input_data, _ in train_loader:
        input_data = input_data.float().to(device)
        output, series, prior, _ = model(input_data)

        # MSE along features
        loss = torch.mean(local_criterion(input_data, output), dim=-1)

        series_loss, prior_loss = 0.0, 0.0
        for idx_u in range(len(prior)):
            normalized_prior = prior[idx_u] / torch.unsqueeze(
                torch.sum(prior[idx_u], dim=-1), dim=-1
            ).repeat(1, 1, 1, win_size)
            sl = my_kl_loss(series[idx_u], normalized_prior.detach()) * temperature
            pl = my_kl_loss(normalized_prior, series[idx_u].detach()) * temperature
            series_loss += sl
            prior_loss += pl

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        train_energy_list.append(cri)

    train_energy = np.concatenate(train_energy_list, axis=0).reshape(-1)

    # (b) get test energy
    test_energy_list = []
    for input_data, _ in thre_loader:
        input_data = input_data.float().to(device)
        output, series, prior, _ = model(input_data)

        # MSE along feature dimension
        loss = torch.mean(local_criterion(input_data, output), dim=-1)

        series_loss, prior_loss = 0.0, 0.0
        for idx_u in range(len(prior)):
            normalized_prior = prior[idx_u] / torch.unsqueeze(
                torch.sum(prior[idx_u], dim=-1), dim=-1
            ).repeat(1, 1, 1, win_size)
            sl = my_kl_loss(series[idx_u], normalized_prior.detach()) * temperature
            pl = my_kl_loss(normalized_prior, series[idx_u].detach()) * temperature
            series_loss += sl
            prior_loss += pl

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        test_energy_list.append(cri)

    test_energy_for_thresh = np.concatenate(test_energy_list, axis=0).reshape(-1)

    # combine
    combined_energy = np.concatenate([train_energy, test_energy_for_thresh], axis=0)
    thresh = np.percentile(combined_energy, 100 - anormly_ratio)

    # (c) final test predictions
    pred_list = []
    test_labels_list = []
    for input_data, labels in thre_loader:
        input_data = input_data.float().to(device)
        labels = labels.detach().cpu().numpy()
        output, series, prior, _ = model(input_data)

        # MSE along features
        loss = torch.mean(local_criterion(input_data, output), dim=-1)

        series_loss, prior_loss = 0.0, 0.0
        for idx_u in range(len(prior)):
            normalized_prior = prior[idx_u] / torch.unsqueeze(
                torch.sum(prior[idx_u], dim=-1), dim=-1
            ).repeat(1, 1, 1, win_size)
            sl = my_kl_loss(series[idx_u], normalized_prior.detach()) * temperature
            pl = my_kl_loss(normalized_prior, series[idx_u].detach()) * temperature
            series_loss += sl
            prior_loss += pl
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = (metric * loss).detach().cpu().numpy()

        pred_list.append(cri)
        test_labels_list.append(labels)

    preds = np.concatenate(pred_list, axis=0).reshape(-1)
    gt = np.concatenate(test_labels_list, axis=0).reshape(-1)

    # final threshold
    y_hat = (preds > thresh).astype(int)
    # detection adjustment
    y_hat = detection_adjustment(y_hat, gt)

    # metrics
    scores = compute_scores(gt, y_hat)

    print("==== AnomalyTransformer Final Results ====")
    print(f"Threshold: {thresh:.6f}")
    print(
        f"Accuracy: {scores['accuracy']:.4f}, Precision: {scores['precision']:.4f}, "
        f"Recall: {scores['recall']:.4f}, F1: {scores['f1_score']:.4f}"
    )

    results = {
        "train_energy_mean": float(train_energy.mean()),
        "train_losses": train_losses,
        "val1_losses": val1_losses,
        "val2_losses": val2_losses,
        "threshold": float(thresh),
        "scores": scores,
    }
    return results, model
