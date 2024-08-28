"""
Download the pre-generated model answers and judgments for MT-bench.
"""

import argparse
import os

from fastchat.utils import run_cmd

filenames = [
    "data/mt_bench_en/model_answer/alpaca-13b.jsonl",
    "data/mt_bench_en/model_answer/baize-v2-13b.jsonl",
    "data/mt_bench_en/model_answer/chatglm-6b.jsonl",
    "data/mt_bench_en/model_answer/claude-instant-v1.jsonl",
    "data/mt_bench_en/model_answer/claude-v1.jsonl",
    "data/mt_bench_en/model_answer/dolly-v2-12b.jsonl",
    "data/mt_bench_en/model_answer/falcon-40b-instruct.jsonl",
    "data/mt_bench_en/model_answer/fastchat-t5-3b.jsonl",
    "data/mt_bench_en/model_answer/gpt-3.5-turbo.jsonl",
    "data/mt_bench_en/model_answer/gpt-4.jsonl",
    "data/mt_bench_en/model_answer/gpt4all-13b-snoozy.jsonl",
    "data/mt_bench_en/model_answer/guanaco-33b.jsonl",
    "data/mt_bench_en/model_answer/guanaco-65b.jsonl",
    "data/mt_bench_en/model_answer/h2ogpt-oasst-open-llama-13b.jsonl",
    "data/mt_bench_en/model_answer/koala-13b.jsonl",
    "data/mt_bench_en/model_answer/llama-13b.jsonl",
    "data/mt_bench_en/model_answer/mpt-30b-chat.jsonl",
    "data/mt_bench_en/model_answer/mpt-30b-instruct.jsonl",
    "data/mt_bench_en/model_answer/mpt-7b-chat.jsonl",
    "data/mt_bench_en/model_answer/nous-hermes-13b.jsonl",
    "data/mt_bench_en/model_answer/oasst-sft-4-pythia-12b.jsonl",
    "data/mt_bench_en/model_answer/oasst-sft-7-llama-30b.jsonl",
    "data/mt_bench_en/model_answer/palm-2-chat-bison-001.jsonl",
    "data/mt_bench_en/model_answer/rwkv-4-raven-14b.jsonl",
    "data/mt_bench_en/model_answer/stablelm-tuned-alpha-7b.jsonl",
    "data/mt_bench_en/model_answer/tulu-30b.jsonl",
    "data/mt_bench_en/model_answer/vicuna-13b-v1.3.jsonl",
    "data/mt_bench_en/model_answer/vicuna-33b-v1.3.jsonl",
    "data/mt_bench_en/model_answer/vicuna-7b-v1.3.jsonl",
    "data/mt_bench_en/model_answer/wizardlm-13b.jsonl",
    "data/mt_bench_en/model_answer/wizardlm-30b.jsonl",
    "data/mt_bench_en/model_judgment/gpt-4_single.jsonl",
    "data/mt_bench_en/model_judgment/gpt-4_pair.jsonl",
]


def download(dump_dir: str = "."):
    prefix = "https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/"

    for name in filenames:
        path = os.path.join(dump_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ret = run_cmd(f"wget -q --show-progress -O {path} {prefix + name}")
        assert ret == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", "-dump", default=".", type=str)
    args = parser.parse_args()

    download(args.dump_dir)
