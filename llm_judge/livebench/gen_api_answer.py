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

import llm_judge  # pylint: disable=unused-import
from llm_judge.livebench.common import (
    DATA_DIR,
    chat_completion_giga,
    load_model_answers,
    load_questions_jsonl,
    load_yaml,
    reorg_answer_file,
)
from llm_judge.livebench.model.model_adapter import get_conversation_template


def get_answer(question: dict, model: str, config: dict, answer_file: str):
    if "required_temperature" in question.keys():
        temperature = max(question["required_temperature"], 0.01)
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
        fout.write(json.dumps(ans, ensure_ascii=False) + "\n")


def run_questions(parallel, questions, model, config, answer_file):
    existing_answer = load_model_answers(Path(answer_file).parent)
    count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = []
        for question in questions:
            if model in existing_answer and question["question_id"] in existing_answer[model]:
                count += 1
                continue
            future = executor.submit(get_answer, question, model, config, answer_file)
            futures.append(future)

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()
    logging.warning("Skipped %s items", count)
    if len(questions) > 0:
        reorg_answer_file(answer_file)


def run_bench(bench_name: str, config_path: str, dump_dir: Optional[str] = None):
    config = load_yaml(config_path)
    devices_config = config["devices"]
    assert isinstance(devices_config, dict), devices_config
    parallel = config["parallel"]
    assert isinstance(parallel, int) and parallel >= 1, parallel
    model = config["devices"]["model"]
    assert isinstance(model, str), model

    list_of_question_files = list((DATA_DIR / f"data/{bench_name}").rglob("*.jsonl"))

    release_set = set(["2024-07-26", "2024-06-24"])
    for question_file in list_of_question_files:
        questions = load_questions_jsonl(str(question_file), release_set, None, None)
        if dump_dir is None:
            answer_file = f"data/{bench_name}/{question_file.parent.name}/model_answer/{model}.jsonl"
        else:
            answer_file = f"{dump_dir}/data/{bench_name}/{question_file.parent.name}/model_answer/{model}.jsonl"

        print(f"Questions from {question_file}")
        print(f"Output to {answer_file}")

        run_questions(
            parallel=parallel, questions=questions, model=model, answer_file=answer_file, config=devices_config
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", "-bench", type=str, default="livebench")
    parser.add_argument("--dump-dir", "-dump", type=str, default=None)
    parser.add_argument("--config", "-cfg", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(filename="gen_api.log", filemode="w", level=logging.DEBUG)

    run_bench(bench_name=args.bench_name, config_path=args.config, dump_dir=args.dump_dir)


if __name__ == "__main__":
    main()
