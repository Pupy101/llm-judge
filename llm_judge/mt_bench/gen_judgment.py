import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
from tqdm import tqdm

import llm_judge  # pylint: disable=unused-import
from llm_judge.mt_bench.common import (
    DATA_DIR,
    NEED_REF_CATS,
    Judge,
    MatchSingle,
    check_data,
    load_judge_prompts,
    load_model_answers,
    load_questions,
    load_unique_judgments,
    load_yaml,
    play_a_match_single,
)


def make_match_single(questions, models, model_answers, judge, ref_answers=None, multi_turn=False) -> List[MatchSingle]:
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for m in models:
            q_id = q["question_id"]
            a = model_answers[m][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                matches.append(MatchSingle(dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn))
            else:
                matches.append(MatchSingle(dict(q), m, a, judge, multi_turn=multi_turn))
    return matches


def make_judge_single(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["single-v1"])
    judges["math"] = Judge(judge_model, judge_prompts["single-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(judge_model, judge_prompts["single-v1-multi-turn"], multi_turn=True)
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["single-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


def run_judge(bench_name: str, config_path: str, dump_dir: Optional[str] = None) -> None:
    question_file = str(DATA_DIR.absolute() / f"{bench_name}/question.jsonl")
    judge_file = str(DATA_DIR.absolute() / "judge_prompts.jsonl")
    ref_answer_dir = str(DATA_DIR.absolute() / f"{bench_name}/reference_answer")

    if dump_dir is None:
        answer_dir = f"data/{bench_name}/model_answer"
    else:
        answer_dir = f"{dump_dir}/{bench_name}/model_answer"

    questions = load_questions(question_file, None, None)
    model_answers = load_model_answers(answer_dir)
    ref_answers = load_model_answers(ref_answer_dir)
    judge_prompts = load_judge_prompts(judge_file)

    config = load_yaml(config_path)
    models = config["compare"]
    judge_config = config["openai"]
    judge_model = judge_config["model"]
    parallel = config["parallel"]

    if dump_dir is None:
        output_file = f"data/{bench_name}/model_judgment/{judge_model}_single.jsonl"
    else:
        output_file = f"{dump_dir}/{bench_name}/model_judgment/{judge_model}_single.jsonl"

    judges = make_judge_single(judge_model, judge_prompts)
    baseline_model = None

    check_data(questions, model_answers, ref_answers, models, judges)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make matches
    matches: List[MatchSingle] = []
    matches += make_match_single(question_default, models, model_answers, judges["default"])
    matches += make_match_single(question_math, models, model_answers, judges["math"], ref_answers)
    matches += make_match_single(question_default, models, model_answers, judges["default-mt"], multi_turn=True)
    matches += make_match_single(question_math, models, model_answers, judges["math-mt"], ref_answers, multi_turn=True)

    match_stat: dict = {}
    match_stat["bench_name"] = bench_name
    match_stat["judge"] = judge_model
    match_stat["baseline"] = baseline_model
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    # Show match stats and prompt enter to continue
    print("Stats:")
    print(json.dumps(match_stat, indent=4))

    unique_judgments = load_unique_judgments(output_file) if os.path.exists(output_file) else set()

    # Play matches
    count = 0
    filtered_matches = []
    for match in matches:
        if (match.question["question_id"], match.model, 2 if match.multi_turn else 1) in unique_judgments:
            count += 1
            continue
        filtered_matches.append(match)
    print("Count skipped already judge examples: ", count)

    if parallel == 1:
        for match in tqdm(filtered_matches):
            if (match.question["question_id"], match.model, 2 if match.multi_turn else 1) in unique_judgments:
                count += 1
                continue
            play_a_match_single(match, config=judge_config, output_file=output_file)
    else:

        def play_a_match_wrapper(match):
            play_a_match_single(match, config=judge_config, output_file=output_file)

        np.random.seed(0)
        np.random.shuffle(filtered_matches)  # type: ignore

        with ThreadPoolExecutor(parallel) as executor:
            for match in tqdm(executor.map(play_a_match_wrapper, filtered_matches), total=len(matches)):
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", "-bench", type=str, default="mt_bench_en")
    parser.add_argument("--dump-dir", "-dump", type=str, default=None)
    parser.add_argument("--config", "-cfg", type=str, required=True)
    args = parser.parse_args()
    logging.basicConfig(filename="gen_judgment.log", filemode="w", level=logging.DEBUG)
    run_judge(bench_name=args.bench_name, config_path=args.config, dump_dir=args.dump_dir)
