import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as ticker


def plot_pareto(json_file, log_scale: bool = False):
    """
    Generates scatter plots showing model performance (F1 scores)
    vs. parameter count (millions) and FLOPs (millions).

    Arguments:
        json_file: Path to the JSON file containing experiment data.
        log_scale: If True, uses a logarithmic scale on the x-axis.
    """

    # 1. Load the JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    # 2. Create a DataFrame from the JSON list
    df = pd.DataFrame(data)

    # Flatten out 'model_metrics' into top-level columns
    df["fvcore_flops"] = df["model_metrics"].apply(lambda x: x["fvcore_flops"])
    df["torchinfo_flops"] = df["model_metrics"].apply(lambda x: x["torchinfo_flops"])
    df["torchinfo_params"] = df["model_metrics"].apply(lambda x: x["torchinfo_params"])

    # Flatten out the 'scores' and 'scores_w_det_adj' from within 'results'
    scores_df = pd.json_normalize(df["results"].apply(lambda x: x["scores"]))
    scores_df.columns = [f"scores_{c}" for c in scores_df.columns]

    scores_w_det_adj_df = pd.json_normalize(
        df["results"].apply(lambda x: x["scores_w_det_adj"])
    )
    scores_w_det_adj_df.columns = [
        f"scores_w_det_adj_{c}" for c in scores_w_det_adj_df.columns
    ]

    df = pd.concat([df, scores_df, scores_w_det_adj_df], axis=1)

    # 3. Create 'true_flops' column (selecting which FLOPs to use depending on model_type)
    df["true_flops"] = df.apply(
        lambda row: row["torchinfo_flops"]
        if row["model_type"] == "LSTMAutoencoder"
        else row["fvcore_flops"],
        axis=1,
    )

    # 4. Create new columns for millions of parameters and millions of FLOPs
    #    (assuming you want to divide by 1e6)
    df["params_m"] = df["torchinfo_params"] / 1e6  # Parameters in millions
    df["flops_m"] = df["true_flops"] / 1e6  # FLOPs in millions

    # Ensure the visualization directory exists
    os.makedirs("./visualization", exist_ok=True)

    # 5. Define plot configurations with these new columns
    #    We'll specify custom x-label and y-label for each scatter.
    plot_configs = [
        {
            "x": "params_m",
            "y": "scores_f1_score",
            "xlabel": "Parameters (M)",
            "ylabel": "F1 Score",
            "filename": "paramsM_vs_f1",
        },
        {
            "x": "params_m",
            "y": "scores_w_det_adj_f1_score",
            "xlabel": "Parameters (M)",
            "ylabel": "F1 Score with Detection Adjustment",
            "filename": "paramsM_vs_adj_f1",
        },
        {
            "x": "flops_m",
            "y": "scores_f1_score",
            "xlabel": "MFLOPs",
            "ylabel": "F1 Score",
            "filename": "mflops_vs_f1",
        },
        {
            "x": "flops_m",
            "y": "scores_w_det_adj_f1_score",
            "xlabel": "MFLOPs",
            "ylabel": "F1 Score with Detection Adjustment",
            "filename": "mflops_vs_adj_f1",
        },
    ]

    # 6. Generate each scatter plot
    for config in plot_configs:
        plt.figure(figsize=(10, 6))

        sns.scatterplot(
            data=df,
            x=config["x"],
            y=config["y"],
            hue="model_type",
            palette="viridis",
            s=100,  # Marker size
        )

        plt.xlabel(config["xlabel"])
        plt.ylabel(config["ylabel"])
        plt.title(f"{config['xlabel']} vs {config['ylabel']}")

        # Apply log scale if requested
        if log_scale:
            plt.xscale("log")
            # More detailed major/minor tick settings for log scale
            plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
            plt.gca().xaxis.set_minor_locator(
                ticker.LogLocator(base=10.0, subs="auto", numticks=30)
            )
        else:
            # Even on a linear scale, allow more fine-grained ticks
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
            plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator())

        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.tight_layout()

        # Save to file
        plt.savefig(f"./visualization/{config['filename']}.pdf")
        plt.close()


# Example usage:
# plot_pareto('experiment_data.json', log_scale=True)
