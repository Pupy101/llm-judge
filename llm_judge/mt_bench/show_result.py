import argparse
import os
from typing import List, Optional

import pandas as pd

from llm_judge.mt_bench.common import load_yaml


def display_result(bench_name: str, config_path: str, dump_dir: Optional[str] = None) -> List[dict]:
    config = load_yaml(config_path)
    openai_config = config["openai"]
    openai_model = openai_config["model"]
    if dump_dir is None:
        judge_dir = f"data/{bench_name}/model_judgment"
    else:
        judge_dir = f"{dump_dir}/{bench_name}/model_judgment"
    input_file = os.path.join(judge_dir, f"{openai_model}_single.jsonl")

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    df = df_all[["model", "score", "turn"]]
    df = df[df["score"] != -1]

    model_list = config["compare"]

    df = df[df["model"].isin(model_list)]

    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    print(df_1.sort_values(by="score", ascending=False))

    output_metrics: List[dict] = []

    for _, row in df_1.iterrows():
        data = row.to_dict()
        data.update(dict(zip(["model", "turn"], row.name)))  # type: ignore
        data.update({"bench": bench_name})
        output_metrics.append(data)

    if bench_name.startswith("mt_bench"):
        print("\n########## Second turn ##########")
        df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
        print(df_2.sort_values(by="score", ascending=False))
        for _, row in df_2.iterrows():
            data = row.to_dict()
            data.update(dict(zip(["model", "turn"], row.name)))  # type: ignore
            data.update({"bench": bench_name})
            output_metrics.append(data)

        print("\n########## Average ##########")
        df_3 = df[["model", "score"]].groupby(["model"]).mean()
        print(df_3.sort_values(by="score", ascending=False))
        for _, row in df_3.iterrows():
            data = row.to_dict()
            data.update(dict(zip(["model", "turn"], [row.name, "overall"])))
            data.update({"bench": bench_name})
            output_metrics.append(data)
    return output_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", "-bench", type=str, default="mt_bench_en")
    parser.add_argument("--config", "-cfg", type=str, required=True)
    parser.add_argument("--dump-dir", "-dump", type=str, default=None)
    args = parser.parse_args()

    display_result_single(bench_name=args.bench_name, config_path=args.config, dump_dir=args.dump_dir)
