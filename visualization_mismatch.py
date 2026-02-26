from os.path import join

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.ticker import MultipleLocator

import helpers


def visualize_bootstrap_scores_vertical(df, ylabel, output_dir: str):
    mpl.rcParams.update(
        {
            # "text.usetex": True,
            # "font.family": "serif",
            # "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 12,
            "font.size": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    # Compute statistics
    bars = pd.DataFrame(
        {
            "lower": df.quantile(0.025),
            "rating": df.quantile(0.5),
            "upper": df.quantile(0.975),
        }
    ).reset_index(names="model")
    bars = bars[~bars["model"].str.endswith("Mismatched")]

    # Error margins
    bars["error_y"] = bars["upper"] - bars["rating"]
    bars["error_y_minus"] = bars["rating"] - bars["lower"]

    # Sort by rating in DESCENDING order
    bars = bars.sort_values("rating", ascending=False).reset_index(drop=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(5, 5))  # Adjust width dynamically

    x = np.arange(len(bars))
    y = bars["rating"]
    yerr = np.vstack([bars["error_y_minus"], bars["error_y"]])

    # Vertical error bars
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="_",
        lw=3,
        capsize=5,
        capthick=3,
        color="blue",
        ecolor="blue",
    )
    ax.axhline(
        0.5,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="chance performance",
    )
    # Add CI labels and rating above each point
    for i, row in bars.iterrows():
        rating_text = f"{row['rating']: .2f}"
        lower_text = f"{row['lower']: .2f}"
        upper_text = f"{row['upper']: .2f}"

        # CI bounds
        ax.text(
            x[i],
            row["lower"] - 0.005,
            lower_text,
            ha="center",
            va="top",
            fontsize=10,
            color="grey",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2),
        )
        ax.text(
            x[i],
            row["upper"] + 0.005,
            upper_text,
            ha="center",
            va="bottom",
            fontsize=10,
            color="grey",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2),
        )

        ax.text(
            x[i] + 0.2,
            row["rating"],
            rating_text,
            ha="left",
            va="center",
            fontsize=10,  # weight='bold',
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2),
        )

    # Aesthetics
    xlabels = [label.replace("BEAT", "Motion capture") for label in bars["model"]]
    to_short = {
        "Motion capture": "mocap",
        "DiffuseStyleGesture": "DSG",
        "SemanticGesticulator": "SG",
        "HoloGest": "HG",
        "AMUSE": "AMUSE",
        "ConvoFusion": "CF",
        "RAG-Gesture": "RG",
    }

    ax.set_xticks(x)

    # 2 display variants of xlabels
    if False:
        xlabels = [to_short[label] for label in xlabels]
        ax.set_xticklabels(xlabels, rotation=0, ha="center", fontsize=12)
    else:
        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=12)

    ax.set_ylabel(ylabel, fontsize=14)
    ylim = (0.4, 0.8)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.5, 7)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    fig.savefig(
        join(output_dir, "mismatching_pref_results_split_ties.png"),
        bbox_inches="tight",
        dpi=1200,
    )
    fig.savefig(
        join(output_dir, "mismatching_pref_results_split_ties.pdf"),
        bbox_inches="tight",
        dpi=1200,
    )


def visualise_p_values(sig_df, title, target_p=0.05):
    fig = px.imshow(
        sig_df,
        color_continuous_scale="OrRd",
        labels={"color": "p-value"},
        title=title,
        height=800,
    )

    for j in range(len(sig_df)):
        for i in range(j + 1, len(sig_df)):
            if sig_df.iloc[i, j] <= target_p:
                fig.add_shape(
                    type="rect",
                    x0=j - 0.5,
                    x1=j + 0.5,
                    y0=i - 0.5,
                    y1=i + 0.5,
                    line=dict(color="black", width=1),
                    xref="x",
                    yref="y",
                    fillcolor="lightgreen",
                )
            else:
                fig.add_shape(
                    type="rect",
                    x0=j - 0.5,
                    x1=j + 0.5,
                    y0=i - 0.5,
                    y1=i + 0.5,
                    line=dict(color="black", width=1),
                    xref="x",
                    yref="y",
                )

            fig.add_annotation(
                text=f"{sig_df.iloc[i, j]:.4f}",
                x=j,
                y=i,
                showarrow=False,
                font=dict(color="black"),
                xref="x",
                yref="y",
            )
    fig.update_layout(xaxis_side="top")
    return fig


def visualize_juice(battles, output_dir: str, combine_normalize=True):
    # TODO: Merge with visualize_juice() for elo?
    mpl.rcParams.update(
        {
            # "text.usetex": True,
            # "font.family": "serif",
            # "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 10,
            "font.size": 9,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 12,
        }
    )

    battles_p = battles.copy()
    battles_p["juice_list"] = battles_p["juiceOptions"].apply(helpers.parse_juice)
    df = battles_p

    def model_and_result(row):
        if "Mismatched" in row["model_a"]:
            model = row["model_b"]
            if row["winner"] == "tie":
                res = "tie"
            elif row["winner"] == "model_a":
                res = "loss"
            elif row["winner"] == "model_b":
                res = "win"
            else:
                raise ValueError()

        else:
            model = row["model_a"]
            if row["winner"] == "tie":
                res = "tie"
            elif row["winner"] == "model_b":
                res = "loss"
            elif row["winner"] == "model_a":
                res = "win"
            else:
                raise ValueError()

        return pd.Series([model, res], index=["model", "result"])

    df[["model", "result"]] = df.apply(model_and_result, axis=1)

    reason_labels = {
        "UnrealisticMotion": "Unrealistic motion",
        "MotionSmoothness": "Smoothness of the motion",
        "AmountIntensityOfMotion": "Amount and Intensity of the motion",
        "RecognisableGestures": "Recognisable gestures",
        "MotionOther": "Other Reason",
        # "BetterFitForEmotion":     "Better fit for emotion",
        # "BetterMatchContentMeaning": "",
        # "EmphasizeCorrectParts":    "Emphasise"
    }

    exploded = df.explode("juice_list")
    exploded["juice_list"] = exploded["juice_list"].replace("", "<no-reason>")
    raw_reasons = sorted(exploded["juice_list"].dropna().unique())
    ordered_first = [r for r in reason_labels if r in raw_reasons]
    ordered_rest = [r for r in raw_reasons if r not in reason_labels]
    reasons = ordered_first + ordered_rest

    # same color for positive and negative
    palette = px.colors.qualitative.Plotly
    color_map = {r: palette[i % len(palette)] for i, r in enumerate(reasons)}

    # wins
    pos_counts = exploded[exploded["result"] == "win"].groupby(["model", "juice_list"]).size().unstack(fill_value=0)

    # losses
    neg_counts = exploded[exploded["result"] == "loss"].groupby(["model", "juice_list"]).size().unstack(fill_value=0)

    pos_tot = pos_counts.sum(axis=1)
    neg_tot = neg_counts.sum(axis=1)
    combined_tot = pos_tot + neg_tot

    if combine_normalize:
        denom = combined_tot.replace(0, 1)
        pos_pct = 1 * pos_counts.div(denom, axis=0)
        neg_pct = -1 * neg_counts.div(denom, axis=0)
    else:
        pos_denom = pos_tot.replace(0, 1)
        neg_denom = neg_tot.replace(0, 1)
        pos_pct = 1 * pos_counts.div(pos_denom, axis=0)
        neg_pct = -1 * neg_counts.div(neg_denom, axis=0)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(9, 8))  # width=9, height=8 in inches

    model_order = [
        "AMUSE",
        "BEAT",
        "ConvoFusion",
        "DiffuseStyleGesture",
        "HoloGest",
        "RAG-Gesture",
        "Seamless",
        "SemanticGesticulator",
    ]
    pos_pct = pos_pct.reindex(index=model_order, columns=reasons, fill_value=0)
    neg_pct = neg_pct.reindex(index=model_order, columns=reasons, fill_value=0)

    n_models = len(model_order)
    x = np.arange(n_models)
    total_groups = len(reasons)
    bar_width = 0.8 / total_groups  # Space bars evenly
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)

    for idx, reason in enumerate(reasons):
        col = color_map[reason]
        friendly = reason_labels.get(reason, reason)

        # Positive bars
        pos_y = [pos_pct.loc[m, reason] for m in model_order]
        ax.bar(x + idx * bar_width, pos_y, width=bar_width, color=col, label=friendly)

        # Negative bars (stacked downward)
        neg_y = [neg_pct.loc[m, reason] for m in model_order]
        ax.bar(x + idx * bar_width, neg_y, width=bar_width, color=col)

    # Customize layout
    ax.set_xticks(x + bar_width * (total_groups - 1) / 2)
    to_short = {
        "BEAT": "mocap",
        "DiffuseStyleGesture": "DSG",
        "SemanticGesticulator": "SG",
        "HoloGest": "HG",
        "AMUSE": "AMUSE",
        "ConvoFusion": "CF",
        "RAG-Gesture": "RG",
        "Seamless": "SN",
    }

    ax.set_xticklabels([to_short[model] for model in model_order], rotation=0)
    ax.set_ylabel("Relative JUICE option vs mismatched")
    ax.set_ylim(-0.201, 0.301)
    # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y/100:.0%}"))

    ax.set_title("Grouped Win/Loss Chart by Reason", color="black")
    ax.legend(loc="upper right")

    # Styling
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.tick_params(colors="black", which="both")
    ax.yaxis.label.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6, which="both")
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    plt.tight_layout()
    fig.savefig(f"{output_dir}/juice_mismatch_results.pdf", bbox_inches="tight", dpi=1200)
    fig.savefig(f"{output_dir}/juice_mismatch_results.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def visualize_preference_ratio(battles, output_dir: str):
    mpl.rcParams.update(
        {
            # "text.usetex": True,
            # "font.family": "serif",
            # "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 12,
            "font.size": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    win_fractions = helpers.win_rate_split_ties(battles)
    models = [model for model in win_fractions.index if "Mismatched" not in model]
    win_fractions = win_fractions[models]

    # Create the bar plot using matplotlib
    fig = plt.figure(figsize=(8, 6))

    plt.bar(models, win_fractions, color="darkblue", edgecolor="black", zorder=3)

    # Add title with larger font, bold and proper alignment
    plt.title(
        "Preference ratio for model outputs over mismatched stimuli",
        fontsize=18,
        fontweight="bold",
        ha="center",
    )

    # Increase font size for tick labels
    plt.xticks(fontsize=12, rotation=15, ha="right")
    plt.yticks(fontsize=12)

    # Add grid lines (major and minor) to improve readability
    plt.grid(
        True,
        which="both",
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        zorder=0,
    )

    # Annotate each bar with its corresponding value for clarity
    for i, v in enumerate(win_fractions):
        plt.text(
            i,
            v + 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.axhline(y=0.5, color="red", linestyle="--", linewidth=2, label="chance performance")
    plt.legend()
    plt.ylim(0, 1)
    # Set tight layout to prevent clipping of labels and titles
    plt.tight_layout()

    fig.savefig(join(output_dir, "preference_ratio_mismatch.png"), dpi=1200)
