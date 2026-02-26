import os

import pandas as pd

import helpers
from visualization_elo import (
    print_pair_comparisons,
    visualize_battle_statistics,
    visualize_bootstrap_scores_2,
    visualize_juice,
    visualize_pairwise_win_fraction,
)

output_dir = "output/elo"
os.makedirs(output_dir, exist_ok=True)

# ------------------
# PREPARE BATTLES
# ------------------
pd.options.display.float_format = "{:.2f}".format
if True:
    # Read the battles.csv file directly
    battles_csv = "battles_elo.csv"
    battles = pd.read_csv(battles_csv)
else:
    # Compute the battles.csv from user study .json data
    STUDY_NAME = [
        "Saturday",
        "Round_2",
        "Round_3",
        "Round_4_HL",
        "Round_5",
        "Final_round",
        "2025-12-Realism_Round_1",
        "2025-12-Realism_Round_2",
        "2025-12-Realism_Round_3",
    ]
    STUDIES_JSON = "studies.json"
    df = helpers.process_and_filter_database_export(STUDIES_JSON, STUDY_NAME, is_mismatch_study=False)
    battles = helpers.convert_to_battle_csv(df, is_mismatch_study=False)
battles = battles.sort_values(["user", "input_code"])
battles.to_csv(os.path.join(output_dir, "battles.csv"), index=False)
battles_no_ties = battles[~battles["winner"].str.contains("tie")]

# ------------------
# COMPUTE ELO SCORES
# ------------------
### Bradley-Terry model - https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
# In the context of LLM evaluation, models can be assumed to be static.
# In this case, we can directly fit the ratings with Bradley-Terry model,
# which produce significantly stable ratings.
# Here we provide an implementation with logistic regression.
elo_mle_ratings = helpers.compute_mle_elo(battles)
df = (
    pd.DataFrame(
        [[n, elo_mle_ratings[n]] for n in elo_mle_ratings.keys()],
        columns=["Model", "Elo rating"],
    )
    .sort_values("Elo rating", ascending=False)
    .reset_index(drop=True)
)
df["Elo rating"] = (df["Elo rating"] + 0.5).astype(int)
df.index = df.index + 1

# ------------------
# VISUALIZATIONS
# ------------------
print("\n=========== ELO ratings ===========")
print(df)
print_pair_comparisons(battles)
visualize_juice(battles, output_dir=output_dir)
visualize_battle_statistics(battles, battles_no_ties, output_dir=output_dir)
visualize_pairwise_win_fraction(battles_no_ties, output_dir=output_dir)

# ------------------
# BOOTSTRAPPING
# ------------------
BOOTSTRAP_ROUNDS = 10000
bootstrap_elo_lu = helpers.bootstrap_elo(battles, BOOTSTRAP_ROUNDS, output_dir=output_dir)
print(bootstrap_elo_lu)
visualize_bootstrap_scores_2(bootstrap_elo_lu, save=True, output_dir=output_dir)
