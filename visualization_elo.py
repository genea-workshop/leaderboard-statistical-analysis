import ast
from os.path import join

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from cycler import cycler
from matplotlib.ticker import MultipleLocator
from plotly.subplots import make_subplots

from helpers import compute_pairwise_win_fraction

PALE_COLORS = [
    "#636efa",
    "#ef553b",
    "#00cc96",
    "#ab63fa",
    "#ffa64d",
]


def visualize_bootstrap_scores_1(df, title):
    # TODO: Ideally merge into one "visualize_bootstrap_scores"
    bars = (
        pd.DataFrame(dict(lower=df.quantile(0.005), rating=df.quantile(0.5), upper=df.quantile(0.995)))
        .reset_index(names="model")
        .sort_values("rating", ascending=False)
    )
    bars["error_y"] = bars["upper"] - bars["rating"]
    bars["error_y_minus"] = bars["rating"] - bars["lower"]
    bars["rating_rounded"] = np.round(bars["rating"], 2)
    fig = px.scatter(bars, x="model", y="rating", error_y="error_y", error_y_minus="error_y_minus", text="rating_rounded", title=title)
    fig.update_layout(xaxis_title="Model", yaxis_title="Rating", height=600)
    return fig


def visualize_bootstrap_scores_2(df, output_dir: str, save=False):
    # TODO: Ideally merge into one "visualize_bootstrap_scores"
    mpl.rcParams.update(
        {
            "axes.labelsize": 10,
            "font.size": 9,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 12,
        }
    )

    # Compute statistics
    bars = pd.DataFrame({"lower": df.quantile(0.025), "rating": df.quantile(0.5), "upper": df.quantile(0.975)}).reset_index(names="model")

    # Error margins
    bars["error_x"] = bars["upper"] - bars["rating"]
    bars["error_x_minus"] = bars["rating"] - bars["lower"]

    # Sort by rating in ASCENDING order
    bars = bars.sort_values("rating", ascending=True).reset_index(drop=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(7, 3))  # Dynamic height

    y = np.arange(len(bars))
    x = bars["rating"]
    xerr = np.vstack([bars["error_x_minus"], bars["error_x"]])

    # Horizontal error bars
    ax.errorbar(x, y, xerr=xerr, fmt="|", lw=3, capsize=5, capthick=3, color="blue", ecolor="blue")

    # Add CI labels below the error bars and ratings
    for i, row in bars.iterrows():
        # Round values to integers
        rating_text = f"{int(round(row['rating']))}"
        lower_text = f"{int(round(row['lower']))}"
        upper_text = f"{int(round(row['upper']))}"

        # Left side: CI lower bound below the error bar
        ax.text(row["lower"] - 3.5, y[i], lower_text, va="center", ha="right", c="grey")

        # Right side: CI upper bound below the error bar
        ax.text(row["upper"] + 3.5, y[i], upper_text, va="center", ha="left", c="grey")

        # Rating below the bar
        ax.text(row["rating"], y[i] - 0.25, rating_text, va="top", ha="center")

    # Aesthetics
    ax.set_yticks(y)
    ylabels = [label.replace("BEAT2", "Motion capture") for label in bars["model"]]
    to_short = {
        "Motion capture": "mocap",
        "DiffuseStyleGesture": "DSG",
        "SemanticGesticulator": "SG",
        "HoloGest": "HG",
        "AMUSE": "AMUSE",
        "ConvoFusion": "CF",
        "RAG-Gesture": "RG",
        "Seamless": "SN",
    }
    ylabels = [to_short[label] for label in ylabels]
    ax.set_yticklabels(ylabels)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    ax.grid(True, axis="x", linestyle="--", alpha=0.6)

    # Set limits
    ax.set_ylim(-1, len(bars))
    ax.set_xlim(600, 1200)

    if save:
        fig.savefig(join(output_dir, "motion_realism_elo_results.png"), bbox_inches="tight", dpi=1200)
        fig.savefig(join(output_dir, "motion_realism_elo_results.pdf"), bbox_inches="tight", dpi=1200)

    return fig


def visualize_bootstrap_scores_3(df, xlabel):
    # TODO: Ideally merge into one "visualize_bootstrap_scores"
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
    bars = pd.DataFrame({"lower": df.quantile(0.025), "rating": df.quantile(0.5), "upper": df.quantile(0.975)}).reset_index(names="model")
    bars = bars[~bars["model"].str.endswith("Mismatched")]

    # Error margins
    bars["error_x"] = bars["upper"] - bars["rating"]
    bars["error_x_minus"] = bars["rating"] - bars["lower"]

    # Sort by rating in ASCENDING order
    bars = bars.sort_values("rating", ascending=True).reset_index(drop=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 3))  # Dynamic height

    y = np.arange(len(bars))
    x = bars["rating"]
    xerr = np.vstack([bars["error_x_minus"], bars["error_x"]])

    # Horizontal error bars
    ax.errorbar(x, y, xerr=xerr, fmt="|", lw=3, capsize=5, capthick=3, color="blue", ecolor="blue")

    # Add CI labels below the error bars and ratings
    for i, row in bars.iterrows():
        # Round values to integers
        rating_text = f"{row['rating']: .2f}"
        lower_text = f"{row['lower']: .2f}"
        upper_text = f"{row['upper']: .2f}"

        # Left side: CI lower bound below the error bar
        ax.text(row["lower"] - 0.01, y[i], lower_text, va="center", ha="right", fontsize=12, c="grey")

        # Right side: CI upper bound below the error bar
        ax.text(row["upper"] + 0.01, y[i], upper_text, va="center", ha="left", fontsize=12, c="grey")

        # Rating below the bar
        ax.text(row["rating"], y[i] - 0.15, rating_text, va="top", ha="center", fontsize=12)

    # Aesthetics
    ax.set_yticks(y)
    ylabels = [label.replace("BEAT", "Motion capture") for label in bars["model"]]
    ax.set_yticklabels(ylabels, fontsize=14)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_xlabel(xlabel, fontsize=14)
    # ax.set_ylabel("Model", fontsize=12)
    # ax.set_title(title, fontsize=14)
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)

    # Set limits
    ax.set_ylim(-1, len(bars))
    ax.set_xlim(0, 1)
    plt.tight_layout()
    return fig


def visualize_pairwise_win_fraction(battles_no_ties, output_dir: str, max_num_models=30):
    ## Pairwise Win Fractions
    # Finally, we can also compute the pairwise win fractions.
    # However, because each model can play as Model A and as Model B
    # and win in both situations we need to compute the wins in both
    # configurations divided by the number of pairings of each model.
    title = "Fraction of Model A Wins for All Non-tied A vs. B Battles"
    row_beats_col = compute_pairwise_win_fraction(battles_no_ties, max_num_models)
    fig = px.imshow(row_beats_col, color_continuous_scale="RdBu", text_auto=".2f", title=title)
    fig.update_layout(xaxis_title=" Model B: Loser", yaxis_title="Model A: Winner", xaxis_side="top", height=900, width=900, title_y=0.07, title_x=0.5)
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(hovertemplate="Model A: %{y}<br>Model B: %{x}<br>Fraction of A Wins: %{z}<extra></extra>")
    pio.write_html(fig, join(output_dir, "pairwise_win_fraction_heatmap.html"))


def visualize_battle_count(battles, title, ordering=None, show_num_models=30):
    ptbl = pd.pivot_table(battles, index="model_a", columns="model_b", aggfunc="size", fill_value=0)
    battle_counts = ptbl + ptbl.T
    if ordering is None:
        ordering = battle_counts.sum().sort_values(ascending=False).index.sort_values(ascending=True)
        ordering = ordering[:show_num_models]

    # Reindex to ensure there are no missing values
    battle_counts = battle_counts.reindex(index=ordering, columns=ordering, fill_value=0)

    fig = px.imshow(battle_counts.loc[ordering, ordering], title=title, text_auto=True)
    fig.update_layout(xaxis_title="Model B", yaxis_title="Model A", xaxis_side="top", title_y=0.95, title_x=0.5, font=dict(size=12))
    fig.update_traces(hovertemplate="Model A: %{y}<br>Model B: %{x}<br>Count: %{z}<extra></extra>")
    return fig, ordering


def visualize_bootstrap_scores_vertical(df):
    # Compute statistics
    bars = pd.DataFrame(
        {
            "lower": df.quantile(0.025),
            "rating": df.quantile(0.5),
            "upper": df.quantile(0.975),
        }
    ).reset_index(names="model")

    # Error margins
    bars["error_y"] = bars["upper"] - bars["rating"]
    bars["error_y_minus"] = bars["rating"] - bars["lower"]

    # Sort by rating in ASCENDING order
    bars = bars.sort_values("rating", ascending=True).reset_index(drop=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(5, 7))  # Wider now

    x = np.arange(len(bars))
    y = bars["rating"]
    yerr = np.vstack([bars["error_y_minus"], bars["error_y"]])

    # Vertical error bars
    ax.errorbar(x, y, yerr=yerr, fmt="o", lw=3, capsize=5, capthick=3, color="blue", ecolor="blue")

    # Add CI labels near the error bars
    for i, row in bars.iterrows():
        rating_text = f"{int(round(row['rating']))}"
        lower_text = f"{int(round(row['lower']))}"
        upper_text = f"{int(round(row['upper']))}"

        # Lower bound
        ax.text(i, row["lower"] - 20, lower_text, va="top", ha="center", c="grey")

        # Upper bound
        ax.text(i, row["upper"] + 20, upper_text, va="bottom", ha="center", c="grey")

        # Rating
        ax.text(i, row["rating"], rating_text, va="bottom", ha="center")

    # X-axis: model names
    xlabels = [label.replace("BEAT2", "Motion capture") for label in bars["model"]]
    to_short = {
        "Motion capture": "mocap",
        "DiffuseStyleGesture": "DSG",
        "SemanticGesticulator": "SG",
        "HoloGest": "HG",
        "AMUSE": "AMUSE",
        "ConvoFusion": "CF",
        "RAG-Gesture": "RG",
    }
    xlabels = [to_short[label] for label in xlabels]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_xlim(-1, len(bars))
    ax.set_ylabel("Elo rating")
    ax.set_title("", fontsize=14)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.set_ylim(600, 1200)
    fig.savefig("output/temp.png")


def visualize_battle_statistics(battles, battles_no_ties, output_dir: str):
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=PALE_COLORS)

    def to_vote(row):
        if row["winner"] == "tie":
            return "tie"
        if row["is_strong"]:
            return row["winner"] + "_" + "clear_pref"
        else:
            return row["winner"] + "_" + "weak_pref"

    battles["vote"] = battles.apply(to_vote, axis=1)

    # --- Battle Outcomes Pie Chart ---
    fig_pie = px.pie(battles["vote"].value_counts().reset_index(), names="vote", values="count", title="Counts of Battle Outcomes", height=400, width=400)
    fig_pie.update_layout(xaxis_title="Battle Outcome", yaxis_title="Count", showlegend=False)
    fig_pie.update_traces(textinfo="label+percent+value")
    pio.write_html(fig_pie, join(output_dir, "battle_outcomes_pie.html"))

    # --- Battle Count Bar Chart ---
    fig_bar = px.bar(pd.concat([battles["model_a"], battles["model_b"]]).value_counts(), title="Battle Count for Each Model", text_auto=True)
    fig_bar.update_layout(xaxis_title="model", yaxis_title="Battle Count", height=400, showlegend=False)
    pio.write_html(fig_bar, join(output_dir, "model_battle_counts_bar.html"))

    # --- Battle Count Heatmaps (Combined) ---
    all_models = sorted(set(battles["model_a"]) | set(battles["model_b"]))
    fig1, _ = visualize_battle_count(battles_no_ties, "Battle Count for Each Combination of Models (without Ties)<br> ", ordering=all_models)
    fig2, _ = visualize_battle_count(battles[battles["winner"].str.contains("tie")], "Tie Count for Each Combination of Models<br> ", ordering=all_models)
    fig_combined = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]], subplot_titles=[fig1.layout.title.text, fig2.layout.title.text])

    # Add traces from fig1 and fig2
    fig_combined.add_trace(fig1.data[0], row=1, col=1)
    fig_combined.add_trace(fig2.data[0], row=1, col=2)
    fig_combined.update_layout(xaxis=dict(side="top"), xaxis2=dict(side="top"))
    fig_combined.update_yaxes(autorange="reversed")
    pio.write_html(fig_combined, join(output_dir, "battle_and_tie_count_heatmaps_combined.html"))

    fig_overall_heatmap, _ = visualize_battle_count(battles, "Total Battle Count for Each Combination of Models")
    pio.write_html(fig_overall_heatmap, join(output_dir, "total_battle_count_heatmap.html"))


def run_statistical_analysis(bootstrap_elo_lu):
    """
    The code is preserved for backup, otherwise it was not found useful and could probably be deleted.
    Ideally broken down further, part of it to go in elo_notebook_helper.py, the visualization stays here.
    """
    elo_scores = {model: np.array(bootstrap_elo_lu[model]) for model in bootstrap_elo_lu.columns}

    elo_diff_samples = {(model_1, model_2): elo_scores[model_1] - elo_scores[model_2] for model_2 in elo_scores.keys() for model_1 in elo_scores.keys()}

    elo_diff_means = {model_pair: diff_samples.mean() for model_pair, diff_samples in elo_diff_samples.items()}

    print(elo_diff_means[("SemanticGesticulator", "ConvoFusion")])

    elo_diff_sample_cis = {
        model_pair: (
            np.percentile(diff_samples, 0.5),
            np.percentile(diff_samples, 99.5),
        )
        for model_pair, diff_samples in elo_diff_samples.items()
    }

    print(elo_diff_sample_cis)
    print(elo_diff_sample_cis[("SemanticGesticulator", "ConvoFusion")])

    def visualise_differences(sig_df, highlight_fn):
        # Plot with Plotly
        fig = px.imshow(
            sig_df,
            color_continuous_scale="OrRd",
            labels={"color": "p-value"},
            title="Significant Elo Differences",
            height=800,
        )

        for j in range(len(sig_df)):
            for i in range(j + 1, len(sig_df)):
                if highlight_fn(sig_df.iloc[i, j]):
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
                    fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5, line=dict(color="black", width=1), xref="x", yref="y")

                fig.add_annotation(text=f"{sig_df.iloc[i, j]:.4f}", x=j, y=i, showarrow=False, font=dict(color="black"), xref="x", yref="y")
        fig.update_layout(xaxis_side="top")
        return fig

    def visualise_diff_cis(sig_df, highlight_fn):
        # Plot with Plotly
        fig = px.imshow(
            sig_df[:, :, 0],
            color_continuous_scale="OrRd",
            labels={"color": "p-value"},
            title="Significant Elo Differences",
            height=800,
        )

        for j in range(len(sig_df)):
            for i in range(j + 1, len(sig_df)):
                if highlight_fn(sig_df[i, j]):
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
                    fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5, line=dict(color="black", width=1), xref="x", yref="y")

                fig.add_annotation(
                    text=f"({sig_df[i, j, 0]:.1f}, {sig_df[i, j, 1]:.1f})", x=j, y=i, showarrow=False, font=dict(color="black"), xref="x", yref="y"
                )
        fig.update_layout(xaxis_side="top")
        return fig

    systems = list(elo_scores.keys())
    n = len(systems)

    difference = np.full((n, n), np.nan)

    diff_ci_array = np.full((n, n, 2), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            diff_ci_array[j, i, :] = elo_diff_sample_cis[(systems[i], systems[j])]
            difference[j, i] = elo_diff_sample_cis[(systems[i], systems[j])][0]

    diff_df = pd.DataFrame(difference, index=systems, columns=systems)
    visualise_diff_cis(diff_ci_array, highlight_fn=lambda x: x[0] > 0).show()
    visualise_differences(diff_df, highlight_fn=lambda x: x > 0).show()

    from matplotlib import pyplot as plt
    from scipy import stats

    bootstrap_scores = bootstrap_elo_lu

    model_scores = {col: bootstrap_scores[col].tolist() for col in bootstrap_scores.columns}
    model_1, model_2 = "ConvoFusion", "HoloGest"

    visualize_bootstrap_scores_2(bootstrap_scores[[model_1, model_2]], "").show()

    print(f"Tested models: {model_1} vs {model_2}")
    a = np.array(model_scores[model_1])
    b = np.array(model_scores[model_2])

    plt.scatter(np.arange(len(a)), (a - b), s=4)
    plt.plot((0, 1000), (0, 0), c="r", lw=2)
    plt.title(f"Elo differences between the two models\nover the course of bootstrapping \nmean={(a - b).mean():.1f},std={(a - b).std():.1f}")
    plt.xlabel("bootstrap round")
    plt.ylabel("elo_A - elo_B")
    print(f"Average Elo scores:\n\t{a.mean():.4f} vs {b.mean():.4f}\n")
    print(f"Sample Elo differences between first 5 bootstraps:\n\t{(a - b)[:5]}\n")
    ttest_result = stats.ttest_rel(a, b, alternative="greater")
    CI = (
        ttest_result.confidence_interval().low,
        ttest_result.confidence_interval().high,
    )
    print(
        f"One-sided T-test result ({model_1} is better than {model_2}):",
        f"p-value: {ttest_result.pvalue}",
        f"95% CI for difference between means:{CI},",
        f"t-stat: {ttest_result.statistic}",
        sep="\n\t",
    )


def print_pair_comparisons(battles):
    print("\n=============== Pair Comparisons ===============")
    df = battles
    df["pair"] = df.apply(lambda row: "-".join(sorted([row["model_a"], row["model_b"]])), axis=1)
    df["pair-clipname"] = df.apply(lambda row: "-".join([row["pair"], row["input_code"]]), axis=1)
    df["participant-pair-clipname"] = df.apply(lambda row: "-".join([row["user"], row["pair"], row["input_code"]]), axis=1)
    print(df["pair"].value_counts())


def visualize_juice(battles, output_dir: str):
    MODEL_ORDER = ["Seamless", "RAG-Gesture", "ConvoFusion", "HoloGest", "SemanticGesticulator", "AMUSE", "DiffuseStyleGesture"]
    BEAT_MODEL = "BEAT2"
    COMBINE_NORMALISE = True  # We could either normalise the win rate by the number of wins/losses or by the number of wins and losses

    reason_labels = {
        "UnrealisticMotion": "Unrealistic motion",
        "MotionSmoothness": "Smoothness of the motion",
        "AmountIntensityOfMotion": "Amount and intensity of the motion",
        "RecognisableGestures": "Recognisable gestures",
        "MotionOther": "Other reason",
    }

    def label(r):
        return reason_labels.get(r, r)

    # remove duplicate entries in battle pd
    battles_p = battles.copy().drop_duplicates(subset=["model_a", "model_b", "winner", "juiceOptions"])

    def parse_juice(val):
        """Return a list even for {}, '', or NaN."""
        if pd.isna(val):
            return []
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            val = val.strip()
            if val == "{}":
                return []
            try:
                parsed = ast.literal_eval(val)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return []

    battles_p["juice_list"] = battles_p["juiceOptions"].apply(parse_juice)

    mask = (battles_p["model_a"] == BEAT_MODEL) | (battles_p["model_b"] == BEAT_MODEL)
    df = battles_p[mask].copy()

    def opponent_and_result(row):
        if row["model_a"] == BEAT_MODEL:
            opp = row["model_b"]
            res = "win" if row["winner"] == "model_b" else "loss" if row["winner"] == "model_a" else "tie"
        else:
            opp = row["model_a"]
            res = "win" if row["winner"] == "model_a" else "loss" if row["winner"] == "model_b" else "tie"
        return pd.Series([opp, res], index=["opponent", "result"])

    df[["opponent", "result"]] = df.apply(opponent_and_result, axis=1)
    df = df[df["opponent"].isin(MODEL_ORDER) & df["result"].isin(["win", "loss"])]

    exploded = df.explode("juice_list")
    exploded["juice_list"] = exploded["juice_list"].replace("", "<no-reason>")
    raw_reasons = sorted(exploded["juice_list"].dropna().unique())
    ordered_first = [r for r in reason_labels if r in raw_reasons]
    ordered_rest = [r for r in raw_reasons if r not in reason_labels]
    reasons = ordered_first + ordered_rest

    # same color for positive and negative
    plt.rcParams["axes.prop_cycle"] = cycler(color=PALE_COLORS)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {r: default_colors[i % len(default_colors)] for i, r in enumerate(reasons)}

    # wins
    pos_counts = exploded[exploded["result"] == "win"].groupby(["opponent", "juice_list"]).size().unstack(fill_value=0)

    # losses
    neg_counts = exploded[exploded["result"] == "loss"].groupby(["opponent", "juice_list"]).size().unstack(fill_value=0)

    pos_tot = pos_counts.sum(axis=1)
    neg_tot = neg_counts.sum(axis=1)
    combined_tot = pos_tot + neg_tot

    if COMBINE_NORMALISE:
        denom = combined_tot.replace(0, 1)
        pos_pct = 1 * pos_counts.div(denom, axis=0)
        neg_pct = -1 * neg_counts.div(denom, axis=0)
    else:
        pos_denom = pos_tot.replace(0, 1)
        neg_denom = neg_tot.replace(0, 1)
        pos_pct = 1 * pos_counts.div(pos_denom, axis=0)
        neg_pct = -1 * neg_counts.div(neg_denom, axis=0)

    mpl.rcParams.update(
        {
            # "font.family": "serif",
            # "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 10,
            "font.size": 9,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 12,
        }
    )

    # Prepare figure
    fig, ax = plt.subplots(figsize=(9, 8))  # width=9, height=8 in inches

    n_models = len(MODEL_ORDER)
    x = np.arange(n_models)
    total_groups = len(reasons)
    bar_width = 0.8 / total_groups  # Space bars evenly
    ax.axhline(0, color="black", linewidth=1.2, linestyle="-")

    for idx, reason in enumerate(reasons):
        col = color_map[reason]
        friendly = label(reason)

        # Positive bars
        pos_y = [pos_pct.loc[m, reason] for m in MODEL_ORDER]
        ax.bar(x + idx * bar_width, pos_y, width=bar_width, color=col, label=friendly)

        # Negative bars (stacked downward)
        neg_y = [neg_pct.loc[m, reason] for m in MODEL_ORDER]
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

    ax.set_xticklabels([to_short[model] for model in MODEL_ORDER], rotation=0)
    ax.set_ylabel("Relative JUICE option vs mismatched")
    ax.set_ylim(-0.25, 0.2)
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
    fig.savefig(join(output_dir, "juice_realism_results.pdf"), bbox_inches="tight", dpi=1200)
    fig.savefig(join(output_dir, "juice_realism_results.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def visualize_average_win_rate(battles_no_ties, save_dir):
    row_beats_col_freq = compute_pairwise_win_fraction(battles_no_ties)
    fig = px.bar(
        row_beats_col_freq.mean(axis=1).sort_values(ascending=False),
        title="Average Win Rate Against All Other Models (Assuming Uniform Sampling and No Ties)",
        text_auto=".2f",
    )
    fig.update_layout(yaxis_title="Average Win Rate", xaxis_title="Model", showlegend=False)
    pio.write_html(fig, f"{save_dir}/average_win_rate.html")
