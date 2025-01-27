import marimo

__generated_with = "0.10.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import subprocess
    import sys
    import torch
    from torchinfo import summary
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    from model.AnomalyTransformer import AnomalyTransformer
    from model.rae import LSTMAutoencoder

    # --- Version and System Info ---
    print("=" * 40, "System Info", "=" * 40)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Get NVIDIA GPU info if available
    try:
        nvidia_info = subprocess.check_output(
            "nvidia-smi", stderr=subprocess.STDOUT
        ).decode()
        print("\nNVIDIA-SMI Output:")
        print(nvidia_info)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("\nNVIDIA-SMI not available")

    # --- GPU Selection ---
    # Set this to choose specific GPU (comma-separated for multiple)

    GPU_ID = 0  # Change this to your desired GPU ID(s)

    # Alternative method using torch
    if torch.cuda.is_available():
        torch.cuda.set_device(GPU_ID)  # Use first specified GPU
        print(
            f"\nUsing GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name()}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1

    RAE_PARAMETERS = {
        "parameters": {
            "window_size": 120,
            "step_size": 40,
            "batch_size": 32,
            "hidden_dim1": 50,
            "hidden_dim2": 100,
            "dropout": 0.2,
            "layer_norm_flag": False,
            "loss_function": "L1Loss",
            "lr": 0.001,
            "num_epochs": 52,
        },
        "min_train_loss": 0.2222,
        "min_val_loss": 0.348,
        "min_train_val_gap": 0.1235,
        "epochs_trained": 44,
        "results_file": "../results/5bfa52f8-e8c6-4899-963d-3ebd80be60f9_history.pkl",
        "timestamp": "2024-04-16 00:52:07.473140",
        "rolling_avg": False,
        "feature_columns": ["ul_bitrate"],
        "dataset_used": "no_outliers_scaled",
    }
    return (
        AnomalyTransformer,
        FlopCountAnalysis,
        GPU_ID,
        LSTMAutoencoder,
        RAE_PARAMETERS,
        batch_size,
        device,
        flop_count_table,
        mo,
        nvidia_info,
        os,
        subprocess,
        summary,
        sys,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""## Util functions""")
    return


@app.cell
def _(FlopCountAnalysis, device, flop_count_table, summary, torch):
    def print_fvcore_flops(model, dummy_input) -> None:
        flops = FlopCountAnalysis(model, dummy_input)
        print(f"Total FLOPs: {flops.total():,}")
        print(flop_count_table(flops))

    def print_torchinfo_summary(model, input_shape, device) -> None:
        summary(
            model,
            input_size=input_shape,
            col_names=[
                "input_size",
                "output_size",
                "num_params",
                "kernel_size",
                "mult_adds",
            ],
            verbose=1,
            device=device,
        )

    def print_metrics(model, input_shape, device) -> None:
        dummy_input = torch.randn(input_shape).to(device)
        print_fvcore_flops(model, dummy_input)
        print_torchinfo_summary(model, input_shape, device)

    def export_to_onnx(model, input_shape, onnx_path) -> None:
        # Export to ONNX
        dummy_input = torch.randn(input_shape).to(device)
        torch.onnx.export(
            model=model,
            args=dummy_input,
            f=onnx_path,
            export_params=True,  # Store trained parameters
            opset_version=15,  # ONNX opset version to use
            do_constant_folding=True,  # Optimize constants
            input_names=["input"],  # Input tensor name
            output_names=["output"],  # Output tensor name
            dynamic_axes={  # Dynamic axes if needed
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        print(f"Model successfully exported to {onnx_path}")

    return (
        export_to_onnx,
        print_fvcore_flops,
        print_metrics,
        print_torchinfo_summary,
    )


@app.cell
def _(mo):
    mo.md(r"""## Initialize on SMD Dataset""")
    return


@app.cell
def _(
    AnomalyTransformer,
    LSTMAutoencoder,
    RAE_PARAMETERS,
    batch_size,
    device,
):
    # Model configuration (defaults from main.py, for SMD Dataset)
    win_size_smd = 100
    enc_in_smd = 38  # Input features
    c_out_smd = 38  # Output features

    # Initialize model and move to device
    anomaly_transformer_model_smd = AnomalyTransformer(
        win_size=win_size_smd, enc_in=enc_in_smd, c_out=c_out_smd
    )

    anomaly_transformer_model_smd.to(device)
    anomaly_transformer_model_smd.eval()  # Ensure model is in eval mode for consistent FLOP calculation

    lstm_auto_priv_model_smd = LSTMAutoencoder(
        input_dim=enc_in_smd,
        hidden_dim1=RAE_PARAMETERS["parameters"]["hidden_dim1"],
        hidden_dim2=RAE_PARAMETERS["parameters"]["hidden_dim2"],
        output_dim=c_out_smd,
        dropout=RAE_PARAMETERS["parameters"]["dropout"],
        layer_norm_flag=RAE_PARAMETERS["parameters"]["layer_norm_flag"],
    )
    lstm_auto_priv_model_smd.to(device)
    lstm_auto_priv_model_smd.eval()

    # Create dummy input matching the model's expected shape (batch_size, win_size, enc_in)
    input_shape_smd = (batch_size, win_size_smd, enc_in_smd)
    return (
        anomaly_transformer_model_smd,
        c_out_smd,
        enc_in_smd,
        input_shape_smd,
        lstm_auto_priv_model_smd,
        win_size_smd,
    )


@app.cell
def _():
    ## Initialize on PRIVATEER Dataset
    return


@app.cell
def _(
    AnomalyTransformer,
    LSTMAutoencoder,
    RAE_PARAMETERS,
    batch_size,
    device,
):
    # Model configuration (defaults from main.py, for SMD Dataset)
    win_size_priv = 120
    enc_in_priv = 1  # Input features
    c_out_priv = 1  # Output features

    # Initialize model and move to device
    anomaly_transformer_model_priv = AnomalyTransformer(
        win_size=win_size_priv, enc_in=enc_in_priv, c_out=c_out_priv
    )
    anomaly_transformer_model_priv.to(device)
    anomaly_transformer_model_priv.eval()  # Ensure model is in eval mode for consistent FLOP calculation

    lstm_auto_priv_model_priv = LSTMAutoencoder(
        input_dim=enc_in_priv,
        hidden_dim1=RAE_PARAMETERS["parameters"]["hidden_dim1"],
        hidden_dim2=RAE_PARAMETERS["parameters"]["hidden_dim2"],
        output_dim=c_out_priv,
        dropout=RAE_PARAMETERS["parameters"]["dropout"],
        layer_norm_flag=RAE_PARAMETERS["parameters"]["layer_norm_flag"],
    )
    lstm_auto_priv_model_priv.to(device)
    lstm_auto_priv_model_priv.eval()

    # Create dummy input matching the model's expected shape (batch_size, win_size, enc_in)
    input_shape_priv = (batch_size, win_size_priv, enc_in_priv)
    return (
        anomaly_transformer_model_priv,
        c_out_priv,
        enc_in_priv,
        input_shape_priv,
        lstm_auto_priv_model_priv,
        win_size_priv,
    )


@app.cell
def _(mo):
    mo.md(r"""## Benchmark SMD""")
    return


@app.cell
def _(
    anomaly_transformer_model_smd,
    device,
    input_shape_smd,
    lstm_auto_priv_model_smd,
    print_metrics,
):
    print(
        "Anomaly Transformer -------------------------------------------------------------"
    )
    print_metrics(anomaly_transformer_model_smd, input_shape_smd, device)
    print(
        "LSTM Autoencoder    -------------------------------------------------------------"
    )
    print_metrics(lstm_auto_priv_model_smd, input_shape_smd, device)
    return


@app.cell
def _(mo):
    mo.md(r"""## Benchmark PRIVATEER Dataset""")
    return


@app.cell
def _(
    anomaly_transformer_model_priv,
    device,
    input_shape_priv,
    lstm_auto_priv_model_priv,
    print_metrics,
):
    print(
        "Anomaly Transformer -------------------------------------------------------------"
    )
    print_metrics(anomaly_transformer_model_priv, input_shape_priv, device)
    print(
        "LSTM Autoencoder    -------------------------------------------------------------"
    )
    print_metrics(lstm_auto_priv_model_priv, input_shape_priv, device)
    return


@app.cell
def _(mo):
    mo.md(r"""## Save to ONNX""")
    return


@app.cell
def _(anomaly_transformer_model_smd, export_to_onnx, input_shape_smd):
    export_to_onnx(
        anomaly_transformer_model_smd,
        input_shape_smd,
        "anomaly_transformer_model_smd.onnx",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Plot Experiments""")
    return


@app.cell
def _(os):
    from experiments.visualizations import plot_pareto

    EXPERIMENTS_JSON_PATH = os.path.join(
        "experiments", "experiments_2025-01-26 17:28:54.json"
    )

    plot_pareto(EXPERIMENTS_JSON_PATH)
    return EXPERIMENTS_JSON_PATH, plot_pareto


if __name__ == "__main__":
    app.run()
