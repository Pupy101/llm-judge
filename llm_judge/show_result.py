import argparse
import os
from typing import Optional

import pandas as pd

from llm_judge.common import load_yaml


def display_result_single(bench_name: str, config_path: str, judge_dir: Optional[str]):
    config = load_yaml(config_path)
    judge_model_config = config["openai"]
    judge_model = judge_model_config["model"]
    if judge_dir is None:
        judge_dir = f"data/{bench_name}/model_judgment"
    input_file = os.path.join(judge_dir, f"{judge_model}_single.jsonl")

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    df = df_all[["model", "score", "turn"]]
    df = df[df["score"] != -1]

    model_list = config["compare"]

    df = df[df["model"].isin(model_list)]

    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    print(df_1.sort_values(by="score", ascending=False))

    if bench_name.startswith("mt_bench"):
        print("\n########## Second turn ##########")
        df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
        print(df_2.sort_values(by="score", ascending=False))

        print("\n########## Average ##########")
        df_3 = df[["model", "score"]].groupby(["model"]).mean()
        print(df_3.sort_values(by="score", ascending=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="mt_bench_en")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--judge-dir", type=str, default=None)
    args = parser.parse_args()

    display_result_single(args.bench_name, args.config, args.judge_dir)
