import argparse
import datetime
import inspect
import math
import os
from collections import defaultdict
from glob import glob
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import llm_judge  # pylint: disable=unused-import
from llm_judge.arena_hard.utils import DATA_DIR, load_model_answers, make_config


def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000, baseline_model="gpt-4"):
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx) // 2 :] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
    lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    # set anchor as gpt-4-0314 = 1000
    if baseline_model in models.index:
        elo_scores += 1000 - elo_scores[models[baseline_model]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def get_bootstrap_result(battles, func_compute_elo, num_round, baseline_model="gpt-4-0314"):
    rows = []
    kwargs = {}
    if baseline_model in inspect.signature(func_compute_elo).parameters:
        kwargs[baseline_model] = baseline_model
    for _ in tqdm(range(num_round), desc="bootstrap"):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True), **kwargs))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def preety_print_two_ratings(ratings_1, ratings_2, column_names):
    df = (
        pd.DataFrame(
            [[n, ratings_1[n], ratings_2[n]] for n in ratings_1.keys()],
            columns=["Model", column_names[0], column_names[1]],
        )
        .sort_values(column_names[0], ascending=False)
        .reset_index(drop=True)
    )
    df[column_names[0]] = (df[column_names[0]] + 0.5).astype(int)
    df[column_names[1]] = (df[column_names[1]] + 0.5).astype(int)
    df.index = df.index + 1
    return df


def visualize_bootstrap_scores(df, title):
    bars = (
        pd.DataFrame(dict(lower=df.quantile(0.025), rating=df.quantile(0.5), upper=df.quantile(0.975)))
        .reset_index(names="model")
        .sort_values("rating", ascending=False)
    )
    bars["error_y"] = bars["upper"] - bars["rating"]
    bars["error_y_minus"] = bars["rating"] - bars["lower"]
    bars["rating_rounded"] = np.round(bars["rating"], 2)
    fig = px.scatter(
        bars,
        x="model",
        y="rating",
        error_y="error_y",
        error_y_minus="error_y_minus",
        text="rating_rounded",
        title=title,
    )
    fig.update_layout(xaxis_title="Model", yaxis_title="Rating", height=600)
    return fig


def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {a: [wins[a][b] if a != b else np.NAN for b in names] for a in names}

    df = pd.DataFrame(data, index=names)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T


def get_win_rate_column(df, column, baseline="gpt-4-0314"):
    to_dict = df[["model", column]].set_index("model").to_dict()[column]
    win_rate_table = predict_win_rate(to_dict)
    return win_rate_table[baseline].fillna(0.5).apply(lambda x: round(x * 100, 2))


def get_battles_from_judgment(judge_name, baseline, dump_dir, bench_name, WEIGHT=3):
    arena_hard_battles = pd.DataFrame()

    print("Turning judgment results into battles...")

    if dump_dir is None:
        directory = f"data/{bench_name}/model_judgment/{judge_name}"
    else:
        directory = f"{dump_dir}/{bench_name}/model_judgment/{judge_name}"

    inner_dir = str(DATA_DIR.absolute() / bench_name / f"model_judgment/{judge_name}")

    assert os.path.exists(directory), directory
    for file in tqdm(glob(f"{directory}/*jsonl") + glob(f"{inner_dir}/*jsonl")):
        df = pd.read_json(file, lines=True)

        for _, row in df.iterrows():
            # game 1
            output = {"question_id": row["question_id"], "model_a": baseline, "model_b": row["model"]}

            game = row["games"][0]

            weight = 1
            if game["score"] == "A=B":
                output["winner"] = "tie"
            elif game["score"] == "A>B":
                output["winner"] = "model_a"
            elif game["score"] == "A>>B":
                output["winner"] = "model_a"
                weight = WEIGHT
            elif game["score"] == "B>A":
                output["winner"] = "model_b"
            elif game["score"] == "B>>A":
                output["winner"] = "model_b"
                weight = WEIGHT
            else:
                weight = 0

            if weight:
                arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])

            output = {"question_id": row["question_id"], "model_a": baseline, "model_b": row["model"]}

            game = row["games"][1]

            weight = 1
            if game["score"] == "A=B":
                output["winner"] = "tie"
            elif game["score"] == "A>B":
                output["winner"] = "model_b"
            elif game["score"] == "A>>B":
                output["winner"] = "model_b"
                weight = WEIGHT
            elif game["score"] == "B>A":
                output["winner"] = "model_a"
            elif game["score"] == "B>>A":
                output["winner"] = "model_a"
                weight = WEIGHT
            else:
                weight = 0

                if weight:
                    arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])
    if dump_dir is None:
        path = f"data/{bench_name}/arena_hard_battles.jsonl"
    else:
        path = f"{dump_dir}/{bench_name}/arena_hard_battles.jsonl"
    arena_hard_battles.to_json(path, lines=True, orient="records")
    return arena_hard_battles


def display_result(
    bench_name: str, config_path: str, baseline: str, dump_dir: Optional[str] = None, num_round: int = 100
) -> List[dict]:
    config = make_config(config_path)
    openai_config = config["openai"]
    judge_model = openai_config["model"]

    model_answers = load_model_answers(str(DATA_DIR / bench_name / "model_answer"))
    if dump_dir is None:
        answer_dir = os.path.join("data", bench_name, "model_answer")
    else:
        answer_dir = os.path.join(dump_dir, bench_name, "model_answer")
    model_answers.update(load_model_answers(answer_dir))

    battles = get_battles_from_judgment(judge_model, baseline=baseline, dump_dir=dump_dir, bench_name=bench_name)

    bootstrap_online_elo = compute_mle_elo(battles, baseline_model=baseline)

    np.random.seed(42)
    bootstrap_elo_lu = get_bootstrap_result(battles, compute_mle_elo, num_round, baseline_model=baseline)
    if dump_dir is None:
        bootsrap_output = "data/bootstrapping_results.jsonl"
    else:
        bootsrap_output = f"{dump_dir}/bootstrapping_results.jsonl"
    bootstrap_elo_lu.to_json(bootsrap_output, lines=True, orient="records")

    stats = pd.DataFrame()
    stats["results"] = None
    stats["results"] = stats["results"].astype("object")

    for i, model in enumerate(bootstrap_online_elo.index):
        assert model in bootstrap_elo_lu.columns

        stats.at[i, "model"] = model
        stats.at[i, "score"] = bootstrap_online_elo[model]  # pylint: disable=unsubscriptable-object
        stats.at[i, "lower"] = np.percentile(bootstrap_elo_lu[model], 2.5)
        stats.at[i, "upper"] = np.percentile(bootstrap_elo_lu[model], 97.5)

        length = 0
        if model in model_answers:
            for _, row in model_answers[model].items():
                turn = row["choices"][0]["turns"][0]
                length += turn["token_len"]
            length /= len(model_answers[model])  # type: ignore

        stats.at[i, "avg_tokens"] = int(length)
        stats.at[i, "results"] = bootstrap_elo_lu[model].tolist()

    stats.sort_values(by="model", inplace=True)
    stats["score"] = get_win_rate_column(stats, "score", baseline=baseline).tolist()
    stats["lower"] = get_win_rate_column(stats, "lower", baseline=baseline).tolist()
    stats["upper"] = get_win_rate_column(stats, "upper", baseline=baseline).tolist()
    decimal = 1

    metrics: List[dict] = []
    stats.sort_values(by="score", ascending=False, inplace=True)
    for _, row in stats.iterrows():
        interval = str((round(row["lower"] - row["score"], decimal), round(row["upper"] - row["score"], decimal)))
        print(
            f"{row['model'] : <30} | score: {round(row['score'], decimal) : ^5} | 95% CI: {interval : ^12} | average #tokens: {int(row['avg_tokens'])}"
        )
        metrics.append(
            {"bench": bench_name, "model": row["model"], "score": row["score"], "tokens": int(row["avg_tokens"])}
        )

    cur_date = datetime.datetime.now()
    date_str = cur_date.strftime("%Y%m%d")
    stats = stats.drop(columns=["results"])
    CI = []
    for i in range(len(stats)):
        score = stats.iloc[i]["score"]
        upper = stats.iloc[i]["upper"]
        lower = stats.iloc[i]["lower"]
        CI.append(f"(-{(score-lower):.2f}, +{(upper-score):.2f})")

    stats["CI"] = CI  # pylint: disable=unsupported-assignment-operation
    col_list = list(stats)
    stats = stats.loc[:, col_list]
    stats.rename(columns={"upper": "rating_q975"}, inplace=True)
    stats.rename(columns={"lower": "rating_q025"}, inplace=True)

    col_list = list(stats)
    col_list[-2], col_list[-1] = col_list[-1], col_list[-2]
    stats = stats.loc[:, col_list]
    stats["date"] = date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:]
    if dump_dir is None:
        stats_path = f"data/arena_hard_leaderboard_{date_str}.csv"
    else:
        stats_path = f"{dump_dir}/arena_hard_leaderboard_{date_str}.csv"
    stats.to_csv(stats_path, index=False)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="arena_hard_en")
    parser.add_argument("--config", "-cfg", type=str, required=True)
    parser.add_argument("--baseline", "-base", type=str, required=True)
    parser.add_argument("--dump-dir", "-dump", type=str, default=None)
    args = parser.parse_args()
    display_result(bench_name=args.bench_name, config_path=args.config, baseline=args.baseline, dump_dir=args.dump_dir)
