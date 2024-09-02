import json
import os
import time
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import Optional

import yaml
from shooters.core.utils import sync_retry_supress
from shooters.devices import sync_chat_completion as giga_sync_chat_completion
from shooters.devices import sync_token as giga_sync_token
from shooters.openai import sync_chat_completion as openai_sync_chat_completion
from shooters.types import DevicesConfig, Message, OpenAIConfig

# API setting constants
GIGACHAT_TOKEN: Optional[str] = None
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

DATA_DIR = Path(__file__).parent / "data"


temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.01,
    "math": 0.01,
    "coding": 0.01,
    "reasoning": 0.01,
    "stem": 0.1,
    "humanities": 0.1,
}


def load_questions(question_file: str):
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


def load_model_answers(answer_dir: str):
    filenames = glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line  # type: ignore
        model_answers[model_name] = answer

    return model_answers


def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


def chat_completion_openai(messages, temperature: float, config: dict) -> str:
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


def chat_completion_giga(messages, temperature: Optional[float], config: dict) -> str:
    global GIGACHAT_TOKEN
    config_: dict = deepcopy(config)  # type: ignore
    params = config_.get("params")
    if params and isinstance(params, dict) and temperature is not None:
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
