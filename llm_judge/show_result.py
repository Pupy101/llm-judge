import argparse
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from llm_judge.common import load_yaml


def display_result_single(bench_name: str, config_path: str, dump_dir: Optional[str]):
    config = load_yaml(config_path)
    judge_model_config = config["openai"]
    judge_model = judge_model_config["model"]
    if dump_dir is None:
        judge_dir = f"data/{bench_name}/model_judgment"
    else:
        judge_dir = f"{dump_dir}/{bench_name}/model_judgment"
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

    filename = "|".join(model_list) + ".jsonl"
    output_path = os.path.join(Path(judge_dir).parent, filename)
    with open(output_path, "w") as fp:
        for _, row in df_1.iterrows():
            data = row.to_dict()
            data.update(dict(zip(["model", "turn"], row.name)))
            fp.write(json.dumps(data, ensure_ascii=False) + "\n")

    if bench_name.startswith("mt_bench"):
        with open(output_path, "a") as fp:
            print("\n########## Second turn ##########")
            df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
            print(df_2.sort_values(by="score", ascending=False))
            for _, row in df_2.iterrows():
                data = row.to_dict()
                data.update(dict(zip(["model", "turn"], row.name)))
                fp.write(json.dumps(data, ensure_ascii=False) + "\n")

            print("\n########## Average ##########")
            df_3 = df[["model", "score"]].groupby(["model"]).mean()
            print(df_3.sort_values(by="score", ascending=False))
            for _, row in df_3.iterrows():
                data = row.to_dict()
                data.update(dict(zip(["model", "turn"], list(row.name) + ["overall"])))
                fp.write(json.dumps(data, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", "-bench", type=str, default="mt_bench_en")
    parser.add_argument("--config", "-cfg", type=str, required=True)
    parser.add_argument("--dump-dir", "-dump", type=str, default=None)
    args = parser.parse_args()

    display_result_single(args.bench_name, args.config, args.dump_dir)
