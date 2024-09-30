import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import nltk
import numpy as np
from tqdm import tqdm

import llm_judge  # pylint: disable=unused-import
from llm_judge.livebench.common import (
    DATA_DIR,
    MatchSingle,
    check_data,
    get_model_list,
    load_model_answers,
    load_questions_jsonl,
    load_yaml,
    make_match_single,
)
from llm_judge.livebench.process_results.coding.utils import LCB_generation_process_results
from llm_judge.livebench.process_results.data_analysis.cta.utils import cta_process_results
from llm_judge.livebench.process_results.data_analysis.tablejoin.utils import joinmap_process_results
from llm_judge.livebench.process_results.data_analysis.tablereformat.utils import table_process_results
from llm_judge.livebench.process_results.instruction_following.utils import instruction_following_process_results
from llm_judge.livebench.process_results.math.AMPS_Hard.utils import amps_hard_process_results
from llm_judge.livebench.process_results.math.math_competitions.utils import (
    aime_process_results,
    mathcontest_process_results,
)
from llm_judge.livebench.process_results.math.olympiad.utils import proof_rearrangement_process_results
from llm_judge.livebench.process_results.reasoning.house_traversal.utils import house_traversal_process_results
from llm_judge.livebench.process_results.reasoning.spatial.utils import spatial_process_results
from llm_judge.livebench.process_results.reasoning.web_of_lies_v2.utils import web_of_lies_process_results
from llm_judge.livebench.process_results.reasoning.zebra_puzzle.utils import zebra_puzzle_process_results
from llm_judge.livebench.process_results.writing.connections.utils import connections_process_results
from llm_judge.livebench.process_results.writing.plot_unscrambling.utils import plot_unscrambling_process_results
from llm_judge.livebench.process_results.writing.typos.utils import typos_process_results


def reorg_output_file(output_file):
    """De-duplicate and sort by question id and model"""
    judgments = {}
    with open(output_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            model = json.loads(l)["model"]
            key = (qid, model)
            judgments[key] = l

    keys = sorted(list(judgments.keys()))
    with open(output_file, "w") as fout:
        for key in keys:
            fout.write(judgments[key])


def play_a_match_gt(match: MatchSingle, output_file: str):
    question, model, answer = (
        match.question,
        match.model,
        match.answer,
    )
    coding_test_case_tasks = ["coding_completion", "LCB_generation"]
    if (
        "ground_truth" not in question
        and "reference" not in question
        and question["task"] not in coding_test_case_tasks
        and question["category"] != "instruction_following"
    ):
        raise ValueError("Questions must have ground_truth to run gen_ground_truth_judgment.")

    task = question["task"]
    task_or_subtask = question["subtask"] if "subtask" in question.keys() else question["task"]
    question_text = question["turns"][0]
    ground_truth = question.get("ground_truth", None)
    llm_answer = answer["choices"][0]["turns"][-1]
    score = 0
    category = None

    # todo: find a better solution than a long if statement.
    if task_or_subtask.split("_")[0] in ["amc", "smc"]:
        score = mathcontest_process_results(ground_truth, llm_answer, question_text)
        category = "math"
    elif task_or_subtask.split("_")[0] == "aime":
        score = aime_process_results(ground_truth, llm_answer)
        category = "math"
    elif task_or_subtask.split("_")[0] in ["imo", "usamo"]:
        score = proof_rearrangement_process_results(ground_truth, llm_answer, edit_distance=True)
        category = "math"
    elif task_or_subtask == "cta":
        score = cta_process_results(ground_truth, llm_answer)
        category = "data_analysis"
    elif task_or_subtask == "tablereformat":
        score = table_process_results(question_text, ground_truth, llm_answer)
        category = "data_analysis"
    elif task_or_subtask == "tablejoin":
        score = joinmap_process_results(question_text, ground_truth, llm_answer)
        category = "data_analysis"
    elif "amps_hard" in task_or_subtask:
        score = amps_hard_process_results(ground_truth, llm_answer)
        category = "math"
    elif task_or_subtask == "web_of_lies_v2":
        score = web_of_lies_process_results(ground_truth, llm_answer)
        category = "reasoning"
    elif task_or_subtask == "house_traversal":
        score = house_traversal_process_results(ground_truth, llm_answer)
        category = "reasoning"
    elif task_or_subtask == "zebra_puzzle":
        score = zebra_puzzle_process_results(ground_truth, llm_answer)
        category = "reasoning"
    elif task_or_subtask == "spatial":
        score = spatial_process_results(ground_truth, llm_answer)
        category = "reasoning"
    elif task_or_subtask == "typos":
        score = typos_process_results(ground_truth, llm_answer)
        category = "language"
    elif task_or_subtask == "connections":
        score = connections_process_results(ground_truth, llm_answer)
        category = "language"
    elif task_or_subtask == "plot_unscrambling":
        score = plot_unscrambling_process_results(ground_truth, llm_answer)
        category = "language"
    elif task_or_subtask in coding_test_case_tasks:
        # use entire question object, because there are test cases inside.
        score = LCB_generation_process_results(question, llm_answer)
        category = "coding"
    else:
        raise NotImplementedError(f"This task ({task_or_subtask}) has not been implemented yet.")

    if not category:
        raise NotImplementedError("A category must be assigned to each task")
    question_id = question["question_id"]
    turn = 1
    result = {
        "question_id": question_id,
        "task": task,
        "model": model,
        "score": score,
        "turn": turn,
        "tstamp": time.time(),
        "category": category,
    }
    if "subtask" in question.keys():
        result["subtask"] = question["subtask"]
    print(f"question: {question_id}, turn: {turn}, model: {model}, " f"score: {score}, ")

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as fout:
            fout.write(json.dumps(result) + "\n")

    return result


def gen_judgments(parallel, questions, output_file, answer_dir, model_list, remove_existing_file, bench_name):

    # Load answers
    model_answers = load_model_answers(answer_dir)

    if model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = model_list

    play_a_match_func = play_a_match_gt

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if output_file and os.path.exists(output_file) and remove_existing_file:
        os.remove(output_file)

    make_match_func = make_match_single
    check_data(questions, model_answers, models)

    # Make matches
    matches = []
    matches += make_match_func(
        questions,
        models,
        model_answers,
    )

    match_stat = {}
    match_stat["bench_name"] = bench_name
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    # Show match stats and prompt enter to continue
    print("Stats:")
    print(json.dumps(match_stat, indent=4))
    # input("Press Enter to confirm...")

    if "instruction_following" in bench_name:
        nltk.download("punkt")
        task_name = matches[0].question["task"]

        if model_list is None:
            models = get_model_list(answer_dir)
        else:
            models = model_list

        for model_id in models:
            scores = instruction_following_process_results(questions, model_answers, task_name, model_id)
            for item in scores:
                question_id = item["question_id"]
                score = item["score"]
                turn = 1
                result = {
                    "question_id": question_id,
                    "task": task_name,
                    "model": model_id,
                    "score": score,
                    "turn": turn,
                    "tstamp": time.time(),
                    "category": "instruction_following",
                }
                print(f"question: {question_id}, turn: {turn}, model: {model_id}, " f"score: {score}, ")

                if output_file:
                    os.makedirs(Path(output_file).parent, exist_ok=True)
                    with open(output_file, "a") as fout:
                        fout.write(json.dumps(result) + "\n")
    else:
        # Play matches
        if parallel == 1:
            for match in tqdm(matches):
                play_a_match_func(match, output_file=output_file)
        else:

            def play_a_match_wrapper(match):
                play_a_match_func(match, output_file=output_file)

            np.random.seed(0)
            np.random.shuffle(matches)

            with ThreadPoolExecutor(parallel) as executor:
                for match in tqdm(executor.map(play_a_match_wrapper, matches), total=len(matches)):
                    pass

    # De-duplicate and sort judgment file
    reorg_output_file(output_file)


def run_judge(bench_name: str, config_path: str, dump_dir: Optional[str] = None):
    config = load_yaml(config_path)
    devices_config = config["devices"]
    assert isinstance(devices_config, dict), devices_config
    parallel = config["parallel"]
    assert isinstance(parallel, int) and parallel >= 1, parallel
    model = config["devices"]["model"]
    assert isinstance(model, str), model
    models = config["compare"]

    list_of_question_files = list((DATA_DIR / f"data/{bench_name}").rglob("*.jsonl"))

    release_set = set(["2024-07-26", "2024-06-24"])
    for question_file in list_of_question_files:
        questions = load_questions_jsonl(str(question_file), release_set, None, None)

        if dump_dir is None:
            answer_dir = f"data/{bench_name}/{question_file.parent.name}/model_answer/"
        else:
            answer_dir = f"{dump_dir}/data/{bench_name}/{question_file.parent.name}/model_answer/"

        if dump_dir is not None:
            output_file = (
                f"{dump_dir}/data/{bench_name}/{question_file.parent.name}/model_judgment/ground_truth_judgment.jsonl"
            )
        else:
            output_file = f"data/{bench_name}/{question_file.parent.name}/model_judgment/ground_truth_judgment.jsonl"

        if len(questions) > 0:
            gen_judgments(
                parallel=parallel,
                questions=questions,
                output_file=output_file,
                answer_dir=answer_dir,
                model_list=models,
                remove_existing_file=False,
                bench_name=question_file.parent.name,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", "-bench", type=str, default="livebench")
    parser.add_argument("--dump-dir", "-dump", type=str, default=None)
    parser.add_argument("--config", "-cfg", type=str, required=True)
    args = parser.parse_args()
    logging.basicConfig(filename="gen_judgment.log", filemode="w", level=logging.DEBUG)
    run_judge(bench_name=args.bench_name, config_path=args.config, dump_dir=args.dump_dir)


if __name__ == "__main__":
    main()
