import argparse
import concurrent.futures
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import shortuuid
import tqdm

from llm_judge.common import (
    QUERY_DIR,
    chat_completion_giga,
    load_questions,
    load_yaml,
    reorg_answer_file,
    temperature_config,
)
from llm_judge.model_adapter import get_conversation_template

logging.basicConfig(filename="gen_api.log", filemode="w", level=logging.DEBUG)


def get_answer(question: dict, model: str, config: dict, answer_file: str):
    if "required_temperature" in question.keys():
        temperature = max(question["required_temperature"], 0.01)
    elif question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    conv = get_conversation_template(model)

    turns = []
    for j in range(len(question["turns"])):
        conv.append_message(conv.roles[0], question["turns"][j])
        conv.append_message(conv.roles[1], None)

        output = chat_completion_giga(conv, temperature, config)

        conv.update_last_message(output)
        turns.append(output)

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": [{"index": 0, "turns": turns}],
        "tstamp": time.time(),
    }

    os.makedirs(Path(answer_file).parent, exist_ok=True)
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans) + "\n")


def run_bench(bench_name: str, dump_dir: Optional[str], config_path: str):
    config = load_yaml(config_path)
    question_file = str(QUERY_DIR / f"{bench_name}/question.jsonl")
    questions = load_questions(question_file, None, None)

    devices_config = config["devices"]
    assert isinstance(devices_config, dict), devices_config
    parallel = config["parallel"]
    assert isinstance(parallel, int) and parallel >= 1, parallel
    model = config["devices"]["model"]
    assert isinstance(model, str), model

    if dump_dir is None:
        answer_file = f"data/{bench_name}/model_answer/{model}.jsonl"
    else:
        answer_file = f"{dump_dir}/{bench_name}/model_answer/{model}.jsonl"
    print(f"Output to {answer_file}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = []
        for question in questions:
            future = executor.submit(get_answer, question, model, devices_config, answer_file)
            futures.append(future)

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()

    reorg_answer_file(answer_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench_en",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--dump-dir", type=str, default=None, help="The output dump dir.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    run_bench(args.bench_name, args.dump_dir, args.config)
