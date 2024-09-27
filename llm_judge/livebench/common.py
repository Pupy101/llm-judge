import dataclasses
import glob
import json
import os
import re
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from datasets import Dataset, load_dataset
from shooters.core.utils import sync_retry_supress
from shooters.devices import sync_chat_completion as giga_sync_chat_completion
from shooters.devices import sync_token as giga_sync_token
from shooters.openai import sync_chat_completion as openai_sync_chat_completion
from shooters.types import DevicesConfig, Message, OpenAIConfig

DATA_DIR = Path(__file__).parent

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


# Extract scores from judgments
two_score_pattern = re.compile("\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]")
two_score_pattern_backup = re.compile("\[(\d+\.?\d*),\s?(\d+\.?\d*)\]")
one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

GIGACHAT_TOKEN: Optional[str] = None

# Huggingface and dataset-related constants
LIVE_BENCH_HF_ORGANIZATION = "livebench"
LIVE_BENCH_DATA_SUPER_PATH = "live_bench"
LIVE_BENCH_CATEGORIES = [
    "coding",
    "data_analysis",
    "instruction_following",
    "math",
    "reasoning",
    "language",
]


@dataclasses.dataclass
class MatchSingle:
    question: dict
    model: str
    answer: dict
    ref_answer: dict = None
    multi_turn: bool = False


def load_yaml(path: str):
    with open(path) as fp:
        config = yaml.safe_load(fp)
    return config


def get_categories_tasks(bench_name: str):
    split_bench_name = bench_name.rstrip("/").split("/")
    assert split_bench_name[0] == "live_bench"
    if len(split_bench_name) == 1:
        # specify entire bench

        categories = {category_name: get_hf_dataset(category_name) for category_name in LIVE_BENCH_CATEGORIES}

        tasks = {
            category_name: get_tasks_from_hf_category(categories[category_name])
            for category_name in LIVE_BENCH_CATEGORIES
        }

    else:
        # specify a category or task
        category_name = split_bench_name[1]

        categories = {category_name: get_hf_dataset(category_name)}

        if len(split_bench_name) == 2:
            tasks = {category_name: get_tasks_from_hf_category(categories[category_name])}
        else:
            assert len(split_bench_name) == 3
            task_name = split_bench_name[2]
            tasks = {category_name: [task_name]}

    return categories, tasks


def get_hf_dataset(dataset_name: str, split="test"):
    return load_dataset(f"{LIVE_BENCH_HF_ORGANIZATION}/{dataset_name}", split=split)


def get_tasks_from_hf_category(category: Dataset):
    return list(set(category["task"]))


def load_answers_judgments():
    model_judgment_dataset = get_hf_dataset("model_judgment", split="leaderboard")
    model_answer_dataset = get_hf_dataset("model_answer", split="leaderboard")

    model_judgment = {
        category_name: [
            example for example in model_judgment_dataset.filter(lambda row: row["category"] == category_name)
        ]
        for category_name in LIVE_BENCH_CATEGORIES
    }

    model_answer = {
        category_name: [
            example for example in model_answer_dataset.filter(lambda row: row["category"] == category_name)
        ]
        for category_name in LIVE_BENCH_CATEGORIES
    }

    return model_answer, model_judgment


def load_questions(
    category: Dataset, livebench_releases: set, task_name: Optional[str], begin: Optional[int], end: Optional[int]
):
    """Load questions from a file."""
    if task_name is not None:
        questions = list(category.filter(lambda row: row["task"] == task_name))
    else:
        questions = list(category)
    questions = questions[begin:end]
    for q in questions:
        if "livebench_release_date" in q.keys() and isinstance(q["livebench_release_date"], datetime):
            q["livebench_release_date"] = datetime.strftime(q["livebench_release_date"], "%Y-%m-%d")
        if "release_date" in q.keys() and isinstance(q["release_date"], datetime):
            q["release_date"] = datetime.strftime(q["release_date"], "%Y-%m-%d")
        if (
            "original_json" in q.keys()
            and "contest_date" in q["original_json"].keys()
            and isinstance(q["original_json"]["contest_date"], datetime)
        ):
            q["original_json"]["contest_date"] = datetime.strftime(
                q["original_json"]["contest_date"], "%Y-%m-%d %H:%M:%S"
            )
    questions = [q for q in questions if q["livebench_release_date"] in livebench_releases]
    return questions


def load_questions_jsonl(question_file: str, livebench_releases: set, begin: Optional[int], end: Optional[int]):
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    questions = [q for q in questions if q["livebench_release_date"] in livebench_releases]
    return questions


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob.glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[: -len(".jsonl")]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    return model_answers


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


def make_match_single(questions, models, model_answers, multi_turn=False):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m = models[i]
            a = model_answers[m][q_id]

            matches.append(MatchSingle(dict(q), m, a, multi_turn=multi_turn))
    return matches


def chat_completion_openai(conv, temperature: float, config: dict):
    config_: dict = deepcopy(config)  # type: ignore
    params = config_.get("params")
    if params and isinstance(params, dict):
        params.update({"temperature": temperature})
    cfg = OpenAIConfig.model_validate(config_)

    assert cfg.api_tokens
    api_key = cfg.api_tokens[0]
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = openai_sync_chat_completion(
                model=cfg.model,
                api_key=api_key,
                params=cfg.params,
                messages=[Message.model_validate(_) for _ in messages],
                base_url=cfg.api.base_url,
                route_chat=cfg.api.route_chat,
            )
            output = response.choices[0].message.content
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output


def chat_completion_giga(conv, temperature: float, config: dict) -> str:
    global GIGACHAT_TOKEN
    config_: dict = deepcopy(config)  # type: ignore
    params = config_.get("params")
    if params and isinstance(params, dict):
        params.update({"temperature": temperature})
    cfg = DevicesConfig.model_validate(config_)

    if GIGACHAT_TOKEN is None and cfg.need_auth:

        @sync_retry_supress
        def token_request() -> str:
            response = giga_sync_token(
                credentials=cfg.credentials,
                scope=cfg.scope,
                auth_url=cfg.api.auth_url,
                request_id=cfg.api.request_id,
            )
            return response.access_token

        GIGACHAT_TOKEN = token_request()
        assert GIGACHAT_TOKEN is not None, "Can't get auth token"

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = giga_sync_chat_completion(
                model=cfg.model,
                token=GIGACHAT_TOKEN,
                params=cfg.params,
                messages=[Message.model_validate(_) for _ in messages],
                base_url=cfg.api.base_url,
                route_chat=cfg.api.route_chat,
            )
            output = response.choices[0].message.content
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output


def load_single_model_judgments(filename: str):
    """Load model judgments.

    The return value is a dict of type:
    Dict[judge: Tuple -> Dict[game_key: tuple -> game_result: dict]
    """
    judge_dict = {}

    for line in open(filename):
        obj = json.loads(line)
        judge = tuple(obj["judge"])
        qid, model = obj["question_id"], obj["model"]

        if judge not in judge_dict:
            judge_dict[judge] = {}

        gamekey = (qid, model)

        judge_dict[judge][gamekey] = {
            "score": obj["score"],
            "judgment": obj["judgment"],
        }
    return judge_dict


def check_data(questions, model_answers, models):
    # check model answers
    for m in models:
        assert m in model_answers, f"Missing model answer for {m}"
        m_answer = model_answers[m]
        for q in questions:
            assert q["question_id"] in m_answer, f"Missing model {m}'s answer to Question {q['question_id']}"


def get_model_list(answer_dir):
    file_paths = glob.glob(f"{answer_dir}/*.jsonl")
    file_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]
    return file_names
