import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from tqdm import tqdm

from llm_judge.common import (
    NEED_REF_CATS,
    QUERY_DIR,
    Judge,
    MatchPair,
    MatchSingle,
    check_data,
    load_judge_prompts,
    load_model_answers,
    load_questions,
    load_yaml,
    play_a_match_single,
)

logging.basicConfig(filename="gen_judgment.log", filemode="w", level=logging.DEBUG)


def make_match(questions, models, model_answers, judge, baseline_model, ref_answers=None, multi_turn=False):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m_1 = models[i]
            m_2 = baseline_model
            if m_1 == m_2:
                continue
            a_1 = model_answers[m_1][q_id]
            a_2 = model_answers[baseline_model][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                match = MatchPair(dict(q), m_1, m_2, a_1, a_2, judge, ref_answer=ref, multi_turn=multi_turn)
            else:
                match = MatchPair(dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn)
            matches.append(match)
    return matches


def make_match_single(questions, models, model_answers, judge, baseline_model=None, ref_answers=None, multi_turn=False):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m = models[i]
            a = model_answers[m][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                matches.append(MatchSingle(dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn))
            else:
                matches.append(MatchSingle(dict(q), m, a, judge, multi_turn=multi_turn))
    return matches


def make_judge_pairwise(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["pair-v2"])
    judges["math"] = Judge(judge_model, judge_prompts["pair-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(judge_model, judge_prompts["pair-v2-multi-turn"], multi_turn=True)
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["pair-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


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


def run_judge(bench_name: str, dump_dir: Optional[str], config_path: str):
    question_file = QUERY_DIR / f"{bench_name}/question.jsonl"
    judge_file = QUERY_DIR / "judge_prompts.jsonl"

    if dump_dir is None:
        answer_dir = f"data/{bench_name}/model_answer"
    else:
        answer_dir = f"{dump_dir}/{bench_name}/model_answer"
    ref_answer_dir = QUERY_DIR / f"{bench_name}/reference_answer"

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
    matches: list = []
    matches += make_match_single(question_default, models, model_answers, judges["default"], baseline_model)
    matches += make_match_single(question_math, models, model_answers, judges["math"], baseline_model, ref_answers)
    matches += make_match_single(
        question_default, models, model_answers, judges["default-mt"], baseline_model, multi_turn=True
    )
    matches += make_match_single(
        question_math, models, model_answers, judges["math-mt"], baseline_model, ref_answers, multi_turn=True
    )

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
    input("Press Enter to confirm...")

    # Play matches
    if parallel == 1:
        for match in tqdm(matches):
            play_a_match_single(match, config=judge_config, output_file=output_file)
    else:

        def play_a_match_wrapper(match):
            play_a_match_single(match, config=judge_config, output_file=output_file)

        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(parallel) as executor:
            for match in tqdm(executor.map(play_a_match_wrapper, matches), total=len(matches)):
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", "-bench", type=str, default="mt_bench_en")
    parser.add_argument("--dump-dir", "-dump", type=str, default=None)
    parser.add_argument("--config", "-cfg", type=str, required=True)
    args = parser.parse_args()
    run_judge(args.bench_name, args.dump_dir, args.config)
