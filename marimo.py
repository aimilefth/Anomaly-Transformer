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
    return (
        AnomalyTransformer,
        FlopCountAnalysis,
        GPU_ID,
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
def _(FlopCountAnalysis, device, flop_count_table, summary):
    def print_fvcore_flops(model, dummy_input) -> None:
        flops = FlopCountAnalysis(model, dummy_input)
        print(f"Total FLOPs: {flops.total():,}")
        print(flop_count_table(flops))

    def print_torchinfo_summary(model, input_shape) -> None:
        print(
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
        )

    return print_fvcore_flops, print_torchinfo_summary


@app.cell
def _(mo):
    mo.md(r"""## Initialize AnomalyTransformer""")
    return


@app.cell
def _(AnomalyTransformer, torch):
    # Model configuration (defaults from main.py, for SMD Dataset)
    win_size = 100
    enc_in = 38  # Input features
    c_out = 38  # Output features
    batch_size = 1

    # Initialize model and move to device
    anomaly_transformer_model = AnomalyTransformer(
        win_size=win_size, enc_in=enc_in, c_out=c_out
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    anomaly_transformer_model.to(device)
    anomaly_transformer_model.eval()  # Ensure model is in eval mode for consistent FLOP calculation

    # Create dummy input matching the model's expected shape (batch_size, win_size, enc_in)

    anomaly_transformer_input_shape = (batch_size, win_size, enc_in)
    anomaly_transformer_dummy_input = torch.randn(anomaly_transformer_input_shape).to(
        device
    )
    return (
        anomaly_transformer_dummy_input,
        anomaly_transformer_input_shape,
        anomaly_transformer_model,
        batch_size,
        c_out,
        device,
        enc_in,
        win_size,
    )


@app.cell
def _(mo):
    mo.md(r"""## Calculate Accurate Flops""")
    return


@app.cell
def _(
    anomaly_transformer_dummy_input,
    anomaly_transformer_model,
    print_fvcore_flops,
):
    # FLOPs analysis using fvcore
    print_fvcore_flops(anomaly_transformer_model, anomaly_transformer_dummy_input)
    return


@app.cell
def _(mo):
    mo.md(r"""## Not so accurate Torchinfo""")
    return


@app.cell
def _(
    anomaly_transformer_input_shape,
    anomaly_transformer_model,
    print_torchinfo_summary,
):
    print_torchinfo_summary(anomaly_transformer_model, anomaly_transformer_input_shape)
    return


if __name__ == "__main__":
    app.run()
