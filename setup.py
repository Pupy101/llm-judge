from pathlib import Path
from typing import List

from setuptools import find_packages, setup


def find_requirements() -> List[str]:
    requirements: List[str] = []
    requirements_file = Path(__file__).parent / "requirement.txt"
    with open(requirements_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                requirements.append(line)
    return sorted(requirements)


setup(
    name="llm_judge",
    version="0.0.2",
    description="llm judge",
    author="Sergei Porkhun",
    author_email="ser.porkhun41@gmail.com",
    packages=find_packages(),
    install_requires=find_requirements(),
    entry_points={
        "console_scripts": [
            "mt_bench_clean_judgment=llm_judge.mt_bench.clean_judgment:main",
            "mt_bench_download_pregenerated=llm_judge.mt_bench.download:main",
            "mt_bench_gen_api_answer=llm_judge.mt_bench.gen_api_answer:main",
            "mt_bench_gen_judgment=llm_judge.mt_bench.gen_judgment:main",
            "mt_bench_show_result=llm_judge.mt_bench.show_result:main",
        ],
    },
    package_data={
        "llm_judge": [
            "arena_hard/data/arena_hard_en/*.jsonl",
            "arena_hard/data/arena_hard_en/model_answer/*.jsonl",
            "arena_hard/data/arena_hard_en/model_judgment/gpt-4o/*.jsonl",
            "arena_hard/data/arena_hard_ru/*.jsonl",
            "arena_hard/data/arena_hard_ru/model_answer/*.jsonl",
            "arena_hard/data/arena_hard_ru/model_judgment/gpt-4o/*.jsonl",
            "mt_bench/data/*.jsonl",
            "mt_bench/data/mt_bench_en/*.jsonl",
            "mt_bench/data/mt_bench_en/reference_answer/*.jsonl",
            "mt_bench/data/mt_bench_ru/*.jsonl",
            "mt_bench/data/mt_bench_ru/reference_answer/*.jsonl",
        ]
    },
)
