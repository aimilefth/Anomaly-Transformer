#!/usr/bin/env python
# experiments/experiments.py

import os
import sys
import json
import uuid
from itertools import product
from typing import Tuple, Dict
import time

import torch

# Make sure we can import from top-level
sys.path.append("..")

from model.AnomalyTransformer import AnomalyTransformer
from model.rae import LSTMAutoencoder
from utils import get_current_timestamp, get_model_metrics
from train_at import train_and_eval_anomaly_transformer
from train_lstm_ae import train_and_eval_lstm_autoencoder


def main():
    ###########################################################################
    # Generate timestamp for the entire experiment run
    ###########################################################################
    run_timestamp = get_current_timestamp()
    ###############################################################################
    # Create directories
    ###############################################################################

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXPERIMENTS_DIR = os.path.join(BASE_DIR, "")
    OUTPUTS_DIR = os.path.join(EXPERIMENTS_DIR, "outputs")
    LOGS_DIR = os.path.join(EXPERIMENTS_DIR, "logs")
    RESULTS_FILE = os.path.join(EXPERIMENTS_DIR, f"experiments_{run_timestamp}.json")

    for d in [OUTPUTS_DIR, LOGS_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
    ###########################################################################
    # Allow GPU selection
    ###########################################################################
    GPU_ID = 0  # <--- Change this to pick your GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(GPU_ID)
        device = "cuda"
        print(f"Using GPU: {GPU_ID} - {torch.cuda.get_device_name(GPU_ID)}")
    else:
        device = "cpu"
        print("GPU not available, running on CPU.")
    ###########################################################################
    # Grid search ranges
    ###########################################################################
    all_experiments = []
    anomaly_transformer_hp = {
        "d_model": [32, 128],
        "n_heads": [2, 4],
        "e_layers": [1, 2],
        "d_ff": [32, 128],
    }
    lstm_auto_hp = {
        "hidden_dim1": [4, 16, 64, 128],
        "hidden_dim2": [4, 16, 64, 128],
    }
    # anomaly_transformer_hp = {
    #     "d_model": [32],
    #     "n_heads": [2],
    #     "e_layers": [1],
    #     "d_ff": [32],
    # }
    # lstm_auto_hp = {
    #     "hidden_dim1": [4],
    #     "hidden_dim2": [4],
    # }

    # Create lists of all combinations
    from itertools import product

    at_combos = list(product(*anomaly_transformer_hp.values()))
    lstm_combos = list(product(*lstm_auto_hp.values()))

    # Calculate total number of runs
    total_runs = len(at_combos) + len(lstm_combos)
    current_run = 1

    ############################################################################
    # 1. Grid search for AnomalyTransformer
    ############################################################################
    print("========== Grid Search: AnomalyTransformer ==========")
    for combo in at_combos:
        d_model, n_heads, e_layers, d_ff = combo
        starting_datetime = get_current_timestamp()
        print(
            f"[{current_run}/{total_runs}] Running AnomalyTransformer with "
            f"d_model={d_model}, n_heads={n_heads}, e_layers={e_layers}, d_ff={d_ff}"
            f"\nStarting at {starting_datetime}"
        )
        start_time = time.time()
        exp_id = str(uuid.uuid4())

        # Instantiate the model to compute metrics (FLOPs, summary, etc.)
        # For SMD dataset, input_c=38, output_c=38, default win_size=100
        tmp_model = AnomalyTransformer(
            win_size=100,
            enc_in=38,
            c_out=38,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=0.0,
            activation="gelu",
            output_attention=True,
        )
        metrics_dict = get_model_metrics(tmp_model, (1, 100, 38), device=device)

        # Train & Evaluate
        results_dict, trained_model = train_and_eval_anomaly_transformer(
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            device=device,
            lr=1e-4,
            k=3.0,
            num_epochs=10,
            batch_size=256,
            anormly_ratio=0.5,
            win_size=100,
            input_c=38,
            output_c=38,
            dataset_name="SMD",
            data_path="../dataset/SMD",
        )
        training_duration = time.time() - start_time
        print(f"Training time: {training_duration} s")
        # Save the trained model
        model_path = os.path.join(OUTPUTS_DIR, f"anomaly_transformer_{exp_id}.pth")
        torch.save(trained_model.state_dict(), model_path)

        # Record the experiment
        experiment_record = {
            "uuid": exp_id,
            "model_type": "AnomalyTransformer",
            "timestamp": starting_datetime,
            "training_duration": float(training_duration),
            "hyperparams": {
                "d_model": d_model,
                "n_heads": n_heads,
                "e_layers": e_layers,
                "d_ff": d_ff,
                "lr": 1e-4,
                "num_epochs": 10,
                "batch_size": 256,
                "k": 3.0,
                "anormly_ratio": 0.5,
            },
            "model_metrics": metrics_dict,
            "results": results_dict,
            "model_path": model_path,
        }

        all_experiments.append(experiment_record)
        # Save the combined experiment records
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_experiments, f, indent=4)
        current_run += 1  # increment run counter

    ############################################################################
    # 2. Grid search for LSTMAutoencoder
    ############################################################################
    print("========== Grid Search: LSTMAutoencoder ==========")
    for combo in lstm_combos:
        hidden_dim1, hidden_dim2 = combo
        starting_datetime = get_current_timestamp()
        print(
            f"[{current_run}/{total_runs}] Running LSTMAutoencoder with "
            f"hidden_dim1={hidden_dim1}, hidden_dim2={hidden_dim2}"
            f"\nStarting at {starting_datetime}"
        )
        start_time = time.time()
        exp_id = str(uuid.uuid4())

        # Metrics
        tmp_model = LSTMAutoencoder(
            input_dim=38,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            output_dim=38,
            dropout=0.0,
            layer_norm_flag=False,
        )
        metrics_dict = get_model_metrics(tmp_model, (1, 100, 38), device=device)

        # Train & Evaluate
        results_dict, trained_model = train_and_eval_lstm_autoencoder(
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            lr=1e-3,
            num_epochs=10,
            batch_size=256,
            anormly_ratio=0.5,
            win_size=100,
            input_c=38,
            output_c=38,
            dataset_name="SMD",
            data_path="../dataset/SMD",
            device=device,
        )
        training_duration = time.time() - start_time
        print(f"Training time: {training_duration} s")
        model_path = os.path.join(OUTPUTS_DIR, f"lstm_autoencoder_{exp_id}.pth")
        torch.save(trained_model.state_dict(), model_path)

        experiment_record = {
            "uuid": exp_id,
            "model_type": "LSTMAutoencoder",
            "timestamp": starting_datetime,
            "training_duration": float(training_duration),
            "hyperparams": {
                "hidden_dim1": hidden_dim1,
                "hidden_dim2": hidden_dim2,
                "lr": 1e-3,
                "num_epochs": 10,
                "batch_size": 256,
                "anormly_ratio": 0.5,
            },
            "model_metrics": metrics_dict,
            "results": results_dict,
            "model_path": model_path,
        }
        all_experiments.append(experiment_record)
        # Save the combined experiment records
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_experiments, f, indent=4)
        current_run += 1  # increment run counter

    # Save the combined experiment records
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_experiments, f, indent=4)
    print(f"\nAll experiments saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
