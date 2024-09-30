import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

import llm_judge  # pylint: disable=unused-import
from llm_judge.livebench.common import DATA_DIR, load_questions_jsonl


def display_result(bench_name: str, config_path: str, dump_dir: Optional[str] = None):

    release_set = set(["2024-07-26", "2024-06-24"])

    directory = f"data/{bench_name}" if dump_dir is None else f"{dump_dir}/data/{bench_name}"

    questions_all = []
    list_of_question_files = list((DATA_DIR / f"data/{bench_name}").rglob("*.jsonl"))

    for question_file in list_of_question_files:
        print(question_file)
        questions = load_questions_jsonl(str(question_file), release_set, None, None)
        questions_all.extend(questions)

    question_id_set = set(q["question_id"] for q in questions_all)

    input_files = [_ for _ in Path(directory).rglob("*.jsonl") if "ground_truth_judgment" == _.stem]
    df_all = pd.concat((pd.read_json(f, lines=True) for f in input_files), ignore_index=True)
    df = df_all[["model", "score", "task", "category", "question_id"]]
    df = df[df["score"] != -1]
    df = df[df["question_id"].isin(question_id_set)]
    df["model"] = df["model"]
    df["score"] *= 100

    model_list_to_check = set(df["model"])
    for model in model_list_to_check:
        df_model = df[df["model"] == model]

        if len(df_model) < len(questions_all):
            raise ValueError(
                f"Invalid result, missing judgments (and possibly completions) for {len(questions_all) - len(df_model)} questions for model {model}."
            )

    print(len(df))

    print("\n########## All Tasks ##########")
    df_1 = df[["model", "score", "task"]]
    df_1 = df_1.groupby(["model", "task"]).mean()
    df_1 = pd.pivot_table(df_1, index=["model"], values="score", columns=["task"], aggfunc="sum")
    df_1 = df_1.round(3)
    print(df_1.sort_values(by="model"))
    df_1.to_csv(f"{directory}/all_tasks.csv")

    print("\n########## All Groups ##########")
    df_1 = df[["model", "score", "category", "task"]]
    df_1 = df_1.groupby(["model", "task", "category"]).mean().groupby(["model", "category"]).mean()
    df_1 = pd.pivot_table(df_1, index=["model"], values="score", columns=["category"], aggfunc="sum")

    df_1["average"] = df_1.mean(axis=1)
    first_col = df_1.pop("average")
    df_1.insert(0, "average", first_col)
    df_1 = df_1.sort_values(by="average", ascending=False)
    df_1 = df_1.round(1)
    print(df_1)
    df_1.to_csv(f"{directory}/all_groups.csv")

    for column in df_1.columns[1:]:
        max_value = df_1[column].max()
        df_1[column] = df_1[column].apply(lambda x: f"\\textbf{{{x}}}" if x == max_value else x)
    df_1.to_csv(f"{directory}/latex_table.csv", sep="&", lineterminator="\\\\\n", quoting=3, escapechar=" ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", "-bench", type=str, default="livebench")
    parser.add_argument("--config", "-cfg", type=str, required=True)
    parser.add_argument("--dump-dir", "-dump", type=str, default=None)
    args = parser.parse_args()

    display_result(bench_name=args.bench_name, config_path=args.config, dump_dir=args.dump_dir)


if __name__ == "__main__":
    main()
