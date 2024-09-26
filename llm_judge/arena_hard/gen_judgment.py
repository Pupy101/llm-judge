import argparse
import concurrent.futures
import json
import logging
import os
import re
from typing import Optional

from tqdm import tqdm

import llm_judge  # pylint: disable=unused-import
from llm_judge.arena_hard.utils import DATA_DIR, chat_completion_openai, load_model_answers, load_questions, make_config

SYSTEM = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."
PROMPT_TEMPLATE = [
    "<|User Prompt|>\n{question_1}\n\n<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>"
]


def get_score(judgment, pattern: re.Pattern, pairwise=True):
    matches = pattern.findall(judgment)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return None, True
    if len(set(matches)) == 1:
        if pairwise:
            return matches[0].strip("\n"), False
        return int(matches[0])
    return None, False


def judgment(**args):
    question = args["question"]
    answer = args["answer"]
    reference = args["reference"]
    baseline = args["baseline_answer"]
    output_file = args["output_file"]
    model = args["judge_model"]
    config = args["config"]

    num_games = 2

    output = {"question_id": question["question_id"], "model": answer["model_id"], "judge": model, "games": []}

    for game in range(num_games):
        conv = [{"role": "system", "content": SYSTEM}]

        for template in PROMPT_TEMPLATE:
            prompt_args = {}

            for i, turn in enumerate(question["turns"]):
                prompt_args[f"question_{i+1}"] = turn["content"]
            base = 1

            if baseline:
                if game % 2 == 1:  # swap position
                    answer, baseline = baseline, answer

                for i, turn in enumerate(baseline["choices"][0]["turns"]):
                    prompt_args[f"answer_{i+1}"] = turn["content"]
                    base += 1
            if answer:
                for i, turn in enumerate(answer["choices"][0]["turns"]):
                    prompt_args[f"answer_{i+base}"] = turn["content"]

            if reference:
                for j, ref_answer in enumerate(reference):
                    for i, turn in enumerate(ref_answer["choices"][0]["turns"]):
                        prompt_args[f"ref_answer_{i+j+1}"] = turn["content"]

            user_prompt = template.format(**prompt_args)
            conv.append({"role": "user", "content": user_prompt})

        judgment = ""
        for _ in range(2):
            new_judgment = chat_completion_openai(conv, 0.1, config)

            judgment += "\n" + new_judgment

            score, try_again = get_score(judgment, args["regex_pattern"])

            conv.append({"role": "assistant", "content": new_judgment})

            if not try_again:
                break

            conv.append(
                {"role": "user", "content": "continue your judgment and finish by outputting a final verdict label"}
            )

        result = {"user_prompt": conv[1]["content"], "judgment": judgment, "score": score}
        output["games"].append(result)

    with open(output_file, "a") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


def run_judge(bench_name: str, config_path: str, baseline: str, dump_dir: Optional[str] = None) -> None:
    config = make_config(config_path)
    openai_config = config["openai"]
    judge_model = openai_config["model"]
    parallel = config["parallel"]

    pattern = re.compile("\[\[([AB<>=]+)\]\]")

    question_file = os.path.join(DATA_DIR, bench_name, "question.jsonl")
    if dump_dir is None:
        answer_dir = os.path.join("data", bench_name, "model_answer")
    else:
        answer_dir = os.path.join(dump_dir, bench_name, "model_answer")

    questions = load_questions(question_file)
    cached_model_answers = load_model_answers(os.path.join(DATA_DIR, bench_name, "model_answer"))
    model_answers = load_model_answers(answer_dir)

    # if user choose a set of models, only judge those models
    models = config["compare"]

    output_files = {}
    if dump_dir is None:
        output_dir = f"data/{bench_name}/model_judgment/{judge_model}"
    else:
        output_dir = f"{dump_dir}/{bench_name}/model_judgment/{judge_model}"

    for model in models:
        output_files[model] = os.path.join(output_dir, f"{model}.jsonl")

    for output_file in output_files.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    existing_judgments = load_model_answers(output_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = []
        for model in models:
            count = 0
            for question in questions:
                question_id = question["question_id"]

                kwargs = {}
                kwargs["question"] = question
                if model in model_answers and not question_id in model_answers[model]:
                    print(f"Warning: {model} answer to {question['question_id']} cannot be found.")
                    continue

                if model in existing_judgments and question_id in existing_judgments[model]:
                    count += 1
                    continue

                kwargs["answer"] = model_answers[model][question_id]
                kwargs["reference"] = None
                assert baseline in cached_model_answers, f"Not found any answers for baseline model: {baseline}"
                kwargs["baseline_answer"] = cached_model_answers[baseline][question_id]
                kwargs["output_file"] = output_files[model]
                kwargs["regex_pattern"] = pattern
                kwargs["judge_model"] = judge_model
                kwargs["config"] = openai_config
                future = executor.submit(judgment, **kwargs)
                futures.append(future)

            if count > 0:
                print(f"{count} number of existing judgments")

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", "-bench", type=str, default="arena_hard_en")
    parser.add_argument("--config", "-cfg", type=str, required=True)
    parser.add_argument("--baseline", "-base", type=str, required=True)
    parser.add_argument("--dump-dir", "-dump", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(filename="gen_judgment.log", filemode="w", level=logging.DEBUG)
    run_judge(bench_name=args.bench_name, config_path=args.config, baseline=args.baseline, dump_dir=args.dump_dir)
