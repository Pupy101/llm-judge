import argparse
import json

selected_models = [
    "alpaca-13b",
    "baize-v2-13b",
    "chatglm-6b",
    "claude-instant-v1",
    "claude-v1",
    "dolly-v2-12b",
    "falcon-40b-instruct",
    "fastchat-t5-3b",
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt4all-13b-snoozy",
    "guanaco-33b",
    "guanaco-65b",
    "h2ogpt-oasst-open-llama-13b",
    "koala-13b",
    "llama-13b",
    "mpt-30b-chat",
    "mpt-30b-instruct",
    "mpt-7b-chat",
    "nous-hermes-13b",
    "oasst-sft-4-pythia-12b",
    "oasst-sft-7-llama-30b",
    "palm-2-chat-bison-001",
    "rwkv-4-raven-14b",
    "stablelm-tuned-alpha-7b",
    "tulu-30b",
    "vicuna-13b-v1.3",
    "vicuna-33b-v1.3",
    "vicuna-7b-v1.3",
    "wizardlm-13b",
    "wizardlm-30b",
]


def clean(infile: str):
    outfile = infile.replace(".jsonl", "_clean.jsonl")

    with open(infile) as fp:
        raw_lines = fp.readlines()
    rets = []
    models = set()
    visited = set()
    for line in raw_lines:
        obj = json.loads(line)

        if "model_1" in obj:  # pair
            model = obj["model_1"]
            key = (
                obj["model_1"],
                obj["model_2"],
                obj["question_id"],
                tuple(obj["judge"]),
            )
        else:  # single
            model = obj["model"]
            key = (obj["model"], obj["question_id"], tuple(obj["judge"]))  # type: ignore

        if key in visited:
            continue
        visited.add(key)

        if model not in selected_models:
            continue
        models.add(model)
        rets.append(obj)

    models = sorted(models)  # type: ignore
    missing_models = [x for x in selected_models if x not in models]
    print(f"in models: {models}, number: {len(models)}")
    print(f"missing models: {missing_models}")
    print(f"#in: {len(raw_lines)}, #out: {len(rets)}")
    rets.sort(
        key=lambda x: (
            x["model"] if "model" in x else x["model_1"],
            x["question_id"],
            x["turn"],
        )
    )

    with open(outfile, "w") as fout:
        for x in rets:
            fout.write(json.dumps(x) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", "-i", type=str)
    args = parser.parse_args()

    clean(args.infile)
