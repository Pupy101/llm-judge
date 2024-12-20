import ast
import dataclasses
import glob
import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import yaml
from shooters.core.utils import sync_retry_supress
from shooters.devices import sync_chat_completion as giga_sync_chat_completion
from shooters.devices import sync_token as giga_sync_token
from shooters.openai import sync_chat_completion as openai_sync_chat_completion
from shooters.types import DevicesConfig, Message, OpenAIConfig

from llm_judge.mt_bench.model_adapter import get_conversation_template

DATA_DIR = Path(__file__).parent / "data"
GIGACHAT_TOKEN: Optional[str] = None

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

TIE_DELTA = 0.1

# Categories that need reference answers
NEED_REF_CATS = ["math", "reasoning", "coding", "arena-hard-200"]

# Extract scores from judgments
two_score_pattern = re.compile("\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]")
two_score_pattern_backup = re.compile("\[(\d+\.?\d*),\s?(\d+\.?\d*)\]")
one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

# Sampling temperature configs for
temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}

reverse_model_map = {
    "model_1": "model_2",
    "model_2": "model_1",
}


@dataclasses.dataclass
class Judge:
    model_name: str
    prompt_template: dict
    ref_based: bool = False
    multi_turn: bool = False


@dataclasses.dataclass
class MatchSingle:
    question: dict
    model: str
    answer: dict
    judge: Judge
    ref_answer: Optional[dict] = None
    multi_turn: bool = False


def load_yaml(path: str):
    with open(path) as fp:
        config = yaml.safe_load(fp)
    return config


def load_jsonl(path: str):
    data = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


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


def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions


def load_model_answers(answer_dir: str):
    filenames = glob.glob(os.path.join(answer_dir, "*.jsonl"))
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


def load_judge_prompts(prompt_file: str):
    prompts = {}
    with open(prompt_file) as fin:
        for line in fin:
            line = json.loads(line)
            prompts[line["name"]] = line  # type: ignore
    return prompts


def run_judge_single(  # pylint: disable=too-many-arguments
    question, answer, judge, ref_answer, config, multi_turn=False
):
    kwargs = {}
    model = judge.model_name
    if ref_answer is not None:
        kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
        if multi_turn:
            kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

    if multi_turn:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question_1=question["turns"][0],
            question_2=question["turns"][1],
            answer_1=answer["choices"][0]["turns"][0],
            answer_2=answer["choices"][0]["turns"][1],
            **kwargs,
        )
    else:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question=question["turns"][0],
            answer=answer["choices"][0]["turns"][0],
            **kwargs,
        )

    rating = -1

    system_prompt = judge.prompt_template["system_prompt"]
    conv = get_conversation_template(model)
    conv.set_system_message(system_prompt)
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    judgment = chat_completion_openai(conv, temperature=0, config=config)

    if judge.prompt_template["output_format"] == "[[rating]]":
        match = re.search(one_score_pattern, judgment)
        if not match:
            match = re.search(one_score_pattern_backup, judgment)

        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = -1
    else:
        raise ValueError(f"invalid output format: {judge.prompt_template['output_format']}")

    return rating, user_prompt, judgment


def play_a_match_single(match: MatchSingle, config, output_file: str):
    question, model, answer, judge, ref_answer, multi_turn = (
        match.question,
        match.model,
        match.answer,
        match.judge,
        match.ref_answer,
        match.multi_turn,
    )

    if judge.prompt_template["type"] == "single":
        score, user_prompt, judgment = run_judge_single(
            question, answer, judge, ref_answer, config, multi_turn=multi_turn
        )

        question_id = question["question_id"]
        turn = 1 if not multi_turn else 2
        result = {
            "question_id": question_id,
            "model": model,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "user_prompt": user_prompt,
            "judgment": judgment,
            "score": score,
            "turn": turn,
            "tstamp": time.time(),
        }
        print(
            f"question: {question_id}, turn: {turn}, model: {model}, "
            f"score: {score}, "
            f"judge: {(judge.model_name, judge.prompt_template['name'])}"
        )
    else:
        raise ValueError(f"invalid judge type: {judge['type']}")  # type: ignore

    if output_file:
        os.makedirs(Path(output_file).parent, exist_ok=True)
        with open(output_file, "a") as fout:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result


def chat_completion_openai(conv, temperature: float, config: dict) -> str:
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


def normalize_game_key_single(gamekey, result):
    qid, model_1, model_2 = gamekey
    if model_1 < model_2:
        return gamekey, result
    new_gamekey = (qid, model_2, model_1)
    new_result = {
        "winners": tuple(reverse_model_map.get(x, x) for x in result["winners"]),
        "g1_judgment": result["g2_judgment"],
        "g2_judgment": result["g1_judgment"],
    }
    return new_gamekey, new_result


def normalize_game_key_dict(judgment_dict):
    ret = {}
    for key, value in judgment_dict.items():
        new_key, new_value = normalize_game_key_single(key, value)
        ret[new_key] = new_value
    return ret


def load_single_model_judgments(filename: str):
    judge_dict: dict = {}

    with open(filename) as fp:
        for line in fp:
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


def load_unique_judgments(filename: str) -> set:
    judgments = set()
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            judgments.add((item["question_id"], item["model"], item["turn"]))
    return judgments


def resolve_single_judgment_dict(question, model_judgments_normal, model_judgments_math, multi_turn=False):
    if multi_turn:
        if question["category"] in NEED_REF_CATS:
            return model_judgments_math[("gpt-4", "single-math-v1-multi-turn")]
        return model_judgments_normal[("gpt-4", "single-v1-multi-turn")]

    if question["category"] in NEED_REF_CATS:
        return model_judgments_math[("gpt-4", "single-math-v1")]
    return model_judgments_normal[("gpt-4", "single-v1")]


def get_single_judge_explanation(gamekey, judgment_dict):
    """Get model judge explanation."""
    try:
        _, model = gamekey

        res = judgment_dict[gamekey]

        g1_judgment = res["judgment"]
        g1_score = res["score"]

        return f"**Game 1**. **A**: {model}, **Score**: {g1_score}\n\n" f"**Judgment**: {g1_judgment}"
    except KeyError:
        return "N/A"


def check_data(questions, model_answers, ref_answers, models, judges):
    # check model answers
    for m in models:
        assert m in model_answers, f"Missing model answer for {m}"
        m_answer = model_answers[m]
        for q in questions:
            assert q["question_id"] in m_answer, f"Missing model {m}'s answer to Question {q['question_id']}"
    # check ref answers
    for jg in judges.values():
        if not jg.ref_based:
            continue
        for q in questions:
            if q["category"] not in NEED_REF_CATS:
                continue
            assert jg.model_name in ref_answers, "Support openai models: " + ",".join(ref_answers.keys())
            assert (
                q["question_id"] in ref_answers[jg.model_name]
            ), f"Missing reference answer to Question {q['question_id']} for judge {jg.model_name}"


def get_model_list(answer_dir):
    file_paths = glob.glob(f"{answer_dir}/*.jsonl")
    file_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]
    return file_names
