# experiments/utils.py

import torch
import numpy as np
import datetime
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torchinfo import summary
from typing import Dict


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


def get_current_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_model_metrics(model, input_shape, device="cpu") -> Dict[str, str]:
    """
    Returns a dictionary with FLOPs table (fvcore) and torchinfo summary.
    Adds torchinfo-specific FLOPs and parameter counts.
    """
    model.to(device)
    model.eval()
    dummy_input = torch.randn(input_shape, device=device)

    # fvcore FLOPs
    flops = FlopCountAnalysis(model, dummy_input)
    flops_table = flop_count_table(flops)

    # torchinfo summary with additional metrics
    model_summary = summary(
        model,
        input_size=input_shape,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds",
        ],
        verbose=0,
        device=device,
    )

    model_summary_str = str(model_summary)

    return {
        "fvcore_flops": flops.total(),
        "fvcore_flops_table": flops_table,
        "torchinfo_summary": model_summary_str,
        "torchinfo_flops": model_summary.total_mult_adds,
        "torchinfo_params": model_summary.total_params,
    }


def compute_scores(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        gt, pred, average="binary"
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f_score),
    }


def detection_adjustment(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Adjust detection in time-series: once an anomaly is detected in an anomalous region,
    label the entire region as anomaly.
    Returns a NEW array instead of modifying in-place.
    """
    adjusted_preds = preds.copy()  # Create a copy of the input array
    anomaly_state = False
    for i in range(len(labels)):
        if labels[i] == 1 and adjusted_preds[i] == 1 and not anomaly_state:
            anomaly_state = True
            # go backwards
            for j in range(i, -1, -1):
                if labels[j] == 0:
                    break
                else:
                    adjusted_preds[j] = 1
            # go forwards
            for j in range(i, len(labels)):
                if labels[j] == 0:
                    break
                else:
                    adjusted_preds[j] = 1
        elif labels[i] == 0:
            anomaly_state = False
        if anomaly_state:
            adjusted_preds[i] = 1
    return adjusted_preds  # Return the modified copy
