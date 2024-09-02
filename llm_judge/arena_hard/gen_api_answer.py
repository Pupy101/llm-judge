import argparse
import concurrent.futures
import json
import logging
import os
import time
from typing import Optional

import shortuuid
import tiktoken
import tqdm

from llm_judge.arena_hard.utils import (
    DATA_DIR,
    chat_completion_giga,
    load_model_answers,
    load_questions,
    make_config,
    reorg_answer_file,
    temperature_config,
)


def get_answer(question: dict, temperature: Optional[float], answer_file: str, config: dict):
    if question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]

    conv = [{"role": "system", "content": "You are a helpful assistant."}]

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    choices = []
    turns = []
    for j in range(len(question["turns"])):
        conv.append({"role": "user", "content": question["turns"][j]["content"]})
        output = chat_completion_giga(messages=conv, temperature=temperature, config=config)
        conv.append({"role": "assistant", "content": output})

        turns.append({"content": output, "token_len": len(encoding.encode(output, disallowed_special=()))})
    choices.append({"index": 0, "turns": turns})

    # Dump answers
    model: str = config["model"]
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans) + "\n")


def run_bench(bench_name: str, config_path: str, dump_dir: Optional[str] = None):
    config = make_config(config_file=config_path)
    devices_config = config["devices"]
    model = devices_config["model"]
    parallel = config["parallel"]

    question_file = os.path.join(DATA_DIR, bench_name, "question.jsonl")
    questions = load_questions(question_file)

    if dump_dir is None:
        existing_files = os.path.join("data", bench_name, "model_answer")
    else:
        existing_files = os.path.join(dump_dir, bench_name, "model_answer")
    existing_answer = load_model_answers(existing_files)

    if dump_dir is None:
        answer_file = os.path.join("data", bench_name, "model_answer", f"{model}.jsonl")
    else:
        answer_file = os.path.join(dump_dir, bench_name, "model_answer", f"{model}.jsonl")

    print(f"Output to {answer_file}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = []
        count = 0
        for question in questions:
            if model in existing_answer and question["question_id"] in existing_answer[model]:
                count += 1
                continue
            future = executor.submit(get_answer, question, None, answer_file, devices_config)
            futures.append(future)
        if count > 0:
            print(f"{count} number of existing answers")
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()

    reorg_answer_file(answer_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", "-bench", type=str, default="arena_hard_en")
    parser.add_argument("--config", "-cfg", type=str, required=True)
    parser.add_argument("--dump-dir", "-dump", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(filename="gen_api.log", filemode="w", level=logging.DEBUG)

    run_bench(bench_name=args.bench_name, config_path=args.config, dump_dir=args.dump_dir)
