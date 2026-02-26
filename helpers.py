import ast
import math
import os
from functools import reduce

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# -------------------------------------------------


def get_system_names(is_mismatch_study):
    system_names_elo = {
        "SW": "DiffuseStyleGesture",
        "SV": "BEAT2",
        "SU": "SemanticGesticulator",
        "ST": "ConvoFusion",
        "SR": "RAG-Gesture",
        "SQ": "HoloGest",
        "SP": "AMUSE",
        "SAH": "Seamless",
    }

    system_names_mismatch = {
        "SX": "DiffuseStyleGesture",
        "SY": "HoloGest",
        "SZ": "SemanticGesticulator",
        "SAA": "RAG-Gesture",
        "SAB": "ConvoFusion",
        "SAC": "BEAT",
        "SAD": "AMUSE",
        "SAG": "Seamless",
    }
    items = list(system_names_mismatch.items())
    for k, v in items:
        system_names_mismatch[k + "_Mismatched"] = v + "_Mismatched"

    system_names = system_names_mismatch if is_mismatch_study else system_names_elo
    return system_names


def process_and_filter_database_export(json_file: str, study_names: list[str] | str, is_mismatch_study: bool):
    print(f"\nProcessing studies.json for studies: {study_names}")

    # Normalize study names
    if isinstance(study_names, str):
        study_names = [study_names]

    # Read studies JSON to dataframe
    with open(json_file) as fp:
        json_file = pd.read_json(fp)
    df = pd.DataFrame(json_file["data"].tolist())

    # Remove entries where the study csv data is missing (this shouldn't happen)
    df = df[df["file_created"].notna()]

    # Filter rows to the current study (defined by the list of study names)
    study_rows = df["file_created"].apply(lambda x: any([x.startswith(study) for study in study_names]))
    df = df[study_rows]

    # Keep only the finished studies
    if is_mismatch_study:
        finished_rows = df["status"] == "finish"
        df = df[finished_rows]

    old_len = len(df)

    # Remove people who fail even a single attention check
    df = df[df["failed_attention_check"] == '"{}"']
    len_diff = old_len - len(df)
    if len_diff > 0:
        print(f"- Removed {len_diff} people who failed attn checks.")

    # Test for duplicated user IDs
    duplicated_user_ids = df[df["prolific_userid"].duplicated(keep=False)][["prolific_userid", "file_created"]]
    if len(duplicated_user_ids) != 0:
        print(f"- Ignoring {len(duplicated_user_ids)} duplicate user IDs in database.")

    duplicated_studies = df[df["file_created"].duplicated(keep=False)][["prolific_userid", "file_created"]]
    if len(duplicated_studies) != 0:
        old_length = len(df)
        df = df.drop_duplicates(subset="file_created", keep="first")
        print(f"- Dropped {old_length - len(df)} duplicate studies.")

    print(f"> {len(df)} studies remain after postprocessing.")

    return df


def who_won_and_is_it_strong_pref(page, is_mismtach_study):
    vote = page["selected"].strip('"')
    if vote == "TheyAreEqual":
        if is_mismtach_study:
            return ("tie", False)
        else:
            return ("tie", None)

    clearly = "Clearly" in vote
    if vote in ["LeftSlightlyBetter", "LeftClearlyBetter"]:
        return "model_a", clearly
    elif vote in ["RightSlightlyBetter", "RightClearlyBetter"]:
        return "model_b", clearly

    raise ValueError(f"Unknown vote {vote} in page {page}")


def convert_to_battle_csv(df, is_mismatch_study):
    print("\nConverting studies data into battles format.")

    vote_database = []
    id_unknown_votes = set()
    id_other_errors = set()
    system_names = get_system_names(is_mismatch_study=is_mismatch_study)

    for _, study in df.iterrows():
        user_id = study["prolific_userid"]

        for page in study["pages"]:
            # Process only the video comparison pages
            if page["type"] != "video":
                continue

            # Sanity check: the two videos in each page should be for the same input sample
            assert page["video1"]["inputcode"] == page["video2"]["inputcode"]

            # Determine the winner and whether it's a strong preference
            try:
                winner, is_strong_pref = who_won_and_is_it_strong_pref(page, is_mismtach_study=is_mismatch_study)
            except ValueError as e:
                if "Unknown vote {} in page" in str(e):
                    vote = str(e).split("Unknown vote ")[1].split(" in page")[0]
                    id_unknown_votes.add((user_id, vote))
                else:
                    id_other_errors.add(user_id)
                continue

            vote_database.append(
                [
                    user_id,
                    page["video1"]["inputcode"],
                    system_names[page["system1"]],
                    system_names[page["system2"]],
                    winner,
                    is_strong_pref,
                    page["juiceOptions"],
                    page["juiceOtherReason"],
                ]
            )

    if len(id_unknown_votes) > 0:
        print("- User IDs with unknown votes:")
        for user_id, vote in id_unknown_votes:
            print(f"--- {user_id} : {vote}")

    if len(id_other_errors) > 0:
        print(f"- User IDs with other errors: {str(id_other_errors)}")

    vote_database = pd.DataFrame(
        vote_database,
        columns=[
            "user",
            "input_code",
            "model_a",
            "model_b",
            "winner",
            "is_strong",
            "juiceOptions",
            "juiceOtherReason",
        ],
    )

    # Optionally filter users, currently kept for dev purposes.
    FILTER_USERS = False
    if FILTER_USERS:
        # users = battles["user"].unique()
        # users = sorted(users.tolist())
        # random.seed(42)
        # random.shuffle(users)
        # N_USERS = 20
        # users = users[:N_USERS]
        users = [
            "5c04872c55614800012b7cf0",
            "5c388683fb3f0f0001fdf65a",
            "67160649a88bd704f98575e9",
            "6553953daddeb31f9794b59c",
            "66dd097265650e0dec23231f",
            "677e3435b16ff7572b1934d2",
            "6507b0bf6c158dca2bfde795",
            "56b78f11e77ebe000cbefe79",
            "60de11d08bb67fc2a1e19c6a",
            "612df482fb26d2d8dab91688",
            "636da58a0d76bbb9167dbf3e",
            "68164650dd74ce8c307b9e7a",
            "6529aca800ead52f4492be19",
            "657a59ca40daebdb26cb31af",
            "6690464404e2d3a5271a608d",
            "6455a2f9877295785cc4a9d3",
            "6741ce8667002c96f0362d34",
            "672fd24711f758e7a04ac043",
            "6691bc059b0182b0bdd7d682",
            "63614a46bd1e8547e0a8b8ba",
        ]
        vote_database = vote_database[vote_database["user"].isin(users)]
        print(f"> Filtered battles to only include these users: {users}")

    print("\nBattle stats:")
    print(f"- Total battles: {len(vote_database)}")
    print(f"- Unique users: {vote_database['user'].nunique()}")
    empty_juice_options = vote_database[vote_database["juiceOptions"] == "{}"]
    print(f"- Battles without JUICE options specified: {len(empty_juice_options)}")
    print("- Winner distribution:")
    winner_counts = vote_database["winner"].value_counts()
    for winner, count in winner_counts.items():
        print(f"--- {winner}: {count}")

    return vote_database


def analyse_counterbalancing(folder, is_mismatch_study):
    df = create_database(folder, is_mismatch_study=is_mismatch_study)
    print("=" * 80)

    if is_mismatch_study:
        list_of_systems = pd.Series(df["system"].unique())
        unique_stimulus_pairs = len(df["system-segment"].unique())
    else:
        list_of_systems = pd.concat([pd.Series(df["system_1"].unique()), pd.Series(df["system_2"].unique())]).unique()
        unique_stimulus_pairs = len(df["pair-clipname"].unique())

    n_systems = len(list_of_systems)
    print(f"[SYSTEMS] {n_systems} systems considered: {sorted(list_of_systems)}")
    print(f"[VOTES] {len(df)} votes in total.")
    print(f"[STIMULUS PAIRS] {unique_stimulus_pairs} unique stimulus pairs used")
    print("=" * 80)

    def summary_text(col):
        counts = col.value_counts()
        return f"{counts.min()} - {counts.max()} (Mean: {counts.mean():.2f} Std: {counts.std():.2f})"

    print(f"[Votes in total] {len(df)}")
    if is_mismatch_study:
        print("[Votes per system] ", summary_text(df["system"]))
        print("[Votes per segment] ", summary_text(df["clip_name"]))
        print("[Votes per comparison] ", summary_text(df["system-segment"]))
        print("=" * 80)
    else:
        print("[Votes per system pair] ", summary_text(df["pair"]))
        print("[Votes per segment] ", summary_text(df["clip_name"]))
        print("[Votes per comparison] ", summary_text(df["pair-clipname"]))
        print(df["pair-clipname"])
        print("=" * 80)
        return df


def create_database(folder, is_mismatch_study):
    df_list = []

    for file in sorted(os.listdir(folder)):
        if file.endswith(".csv"):
            file_path = os.path.join(folder, file)
            temp_df = pd.read_csv(file_path)
            temp_df["participant_number"] = file.split("_")[-1].strip(".csv")  # Add filename as new column
            df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)

    if is_mismatch_study:
        # df['pair'] = df.apply(lambda row: '-'.join(sorted([row['system_1'], row['system_2']])), axis=1)
        df["system-segment"] = df.apply(lambda row: "-".join([row["system"], row["clip_name"]]), axis=1)
        df["participant-sys-clipname"] = df.apply(
            lambda row: "-".join([row["participant_number"], row["system"], row["clip_name"]]),
            axis=1,
        )
    else:
        df["pair"] = df.apply(lambda row: "-".join(sorted([row["system_1"], row["system_2"]])), axis=1)
        df["pair-clipname"] = df.apply(lambda row: "-".join([row["pair"], row["clip_name"]]), axis=1)
        df["participant-pair-clipname"] = df.apply(
            lambda row: "-".join([row["participant_number"], row["pair"], row["clip_name"]]),
            axis=1,
        )

    return df


def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
    # Weak win
    ptbl_a_win_weak = pd.pivot_table(
        df[(df["winner"] == "model_a") & (df["is_strong"] == False)],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    # Strong win
    ptbl_a_win_strong = pd.pivot_table(
        df[(df["winner"] == "model_a") & (df["is_strong"] == True)],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    # if no tie, create a zero matrix
    if sum(df["winner"].isin(["tie", "tie (bothbad)"])) == 0:
        ptbl_tie = pd.DataFrame(0, index=ptbl_a_win_weak.index, columns=ptbl_a_win_weak.columns)
    else:
        ptbl_tie = pd.pivot_table(
            df[df["winner"].isin(["tie", "tie (bothbad)"])],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
        ptbl_tie = ptbl_tie + ptbl_tie.T

    # Weak win
    ptbl_b_win_weak = pd.pivot_table(
        df[(df["winner"] == "model_b") & (df["is_strong"] == False)],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    # Strong win
    ptbl_b_win_strong = pd.pivot_table(
        df[(df["winner"] == "model_b") & (df["is_strong"] == True)],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    ptbl_win = ptbl_a_win_strong * 4 + ptbl_a_win_weak * 2 + ptbl_tie + ptbl_b_win_weak.T * 2 + ptbl_b_win_strong.T * 4
    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            # if nan skip
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = math.log(BASE)
            X[cur_row + 1, models[m_b]] = -math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def compute_pairwise_win_fraction(battles, max_num_models=30):
    # Times each model wins as Model A
    a_win_ptbl = pd.pivot_table(
        battles[battles["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    # Table counting times each model wins as Model B
    b_win_ptbl = pd.pivot_table(
        battles[battles["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    # Table counting number of A-B pairs
    num_battles_ptbl = pd.pivot_table(battles, index="model_a", columns="model_b", aggfunc="size", fill_value=0)

    # Computing the proportion of wins for each model as A and as B
    # against all other models
    row_beats_col_freq = (a_win_ptbl + b_win_ptbl.T) / (num_battles_ptbl + num_battles_ptbl.T)

    # Arrange ordering according to proprition of wins
    prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
    prop_wins = prop_wins[:max_num_models]
    model_names = list(prop_wins.keys())
    row_beats_col = row_beats_col_freq.loc[model_names, model_names]
    return row_beats_col


def bootstrap_elo(battles, bootstrap_rounds, output_dir: str, bootstrap_users=True, seed=42):
    ## Compute Bootstrap Confidence Interavals for BT Scores
    # We can further use bootstrap to estimate the confidence intervals as well.
    np.random.seed(seed)
    rows = []
    if bootstrap_users:
        # Pre-group all user data once
        user_groups_dict = dict(tuple(battles.groupby("user")))
        users = list(user_groups_dict.keys())

        for _ in tqdm(range(bootstrap_rounds), desc="bootstrap"):
            if bootstrap_users:
                # Sample user IDs (not DataFrame)
                sampled_users = np.random.choice(users, size=len(users), replace=True)

                # Collect all rows via list comprehension (avoid concat in loop)
                sampled_dfs = [user_groups_dict[u] for u in sampled_users]
                battle_sample = pd.concat(sampled_dfs, ignore_index=True)
            else:
                battle_sample = battles.sample(frac=1.0, replace=True)

            # Append result of the scoring function
            rows.append(compute_mle_elo(battle_sample))

    # Return columns sorted by median value
    df = pd.DataFrame(rows)
    bootstrap_elo_lu = df[df.median().sort_values(ascending=False).index]
    file_path = os.path.join(output_dir, "bootstrap_elo_lu.csv")
    bootstrap_elo_lu.to_csv(file_path)
    return bootstrap_elo_lu


def bootstrap_mismatch(battles, bootstrap_rounds, bootstrap_users=True, seed=42):
    np.random.seed(seed)
    rows = []
    if bootstrap_users:
        # Pre-group all user data once
        user_groups_dict = dict(tuple(battles.groupby("user")))
        users = list(user_groups_dict.keys())

    for _ in tqdm(range(bootstrap_rounds), desc="bootstrap"):
        if bootstrap_users:
            # Sample user IDs (not DataFrame)
            sampled_users = np.random.choice(users, size=len(users), replace=True)

            # Collect all rows via list comprehension (avoid concat in loop)
            sampled_dfs = [user_groups_dict[u] for u in sampled_users]
            battle_sample = pd.concat(sampled_dfs, ignore_index=True)
        else:
            battle_sample = battles.sample(frac=1.0, replace=True)

        # Append result of the scoring function
        rows.append(win_rate_split_ties(battle_sample))

    df = pd.DataFrame(rows)

    # Return columns sorted by median value
    return df[df.median().sort_values(ascending=False).index]


# from mismatch notebook
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
        except ImportError:
            return []
    return []


def win_rate_split_ties(df):
    """
    Compute the win rate, with ties counting as half wins, and strong preferences counting twice.
    """
    tie_df = df[df["winner"] == "tie"]
    weak_pref_df = df[(df["winner"] != "tie") & (df["is_strong"] == False)]
    strong_pref_df = df[(df["winner"] != "tie") & (df["is_strong"] == True)]

    # Winning outcomes
    win_counts = [
        # Weak wins per model, either as model_a, or as model_b
        weak_pref_df[weak_pref_df["winner"] == "model_a"]["model_a"].value_counts(),
        weak_pref_df[weak_pref_df["winner"] == "model_b"]["model_b"].value_counts(),
        # Strong wins per model, either as model_a, or as model_b -- these count twice.
        2 * strong_pref_df[strong_pref_df["winner"] == "model_a"]["model_a"].value_counts(),
        2 * strong_pref_df[strong_pref_df["winner"] == "model_b"]["model_b"].value_counts(),
        # Ties, either as model_a, or as model_B -- these count half.
        tie_df["model_a"].value_counts() / 2,
        tie_df["model_b"].value_counts() / 2,
    ]

    appearance_counts = [
        # Appearances in weak pref votes
        weak_pref_df["model_a"].value_counts(),
        weak_pref_df["model_b"].value_counts(),
        # # Appearances in strong pref votes -- these count twice.
        2 * strong_pref_df["model_a"].value_counts(),
        2 * strong_pref_df["model_b"].value_counts(),
        # Ties, either as model_a, or as model_B -- these count as half
        tie_df["model_a"].value_counts(),
        tie_df["model_b"].value_counts(),
    ]

    total_wins = reduce(lambda a, b: a.add(b, fill_value=0), win_counts)
    total_appearances = reduce(lambda a, b: a.add(b, fill_value=0), appearance_counts)

    # Compute win rate
    win_rate = total_wins / total_appearances
    return win_rate.sort_values(ascending=False)
