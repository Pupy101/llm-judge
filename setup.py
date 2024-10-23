from pathlib import Path
from typing import List

from setuptools import find_packages, setup

from llm_judge import __version__


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
    version=__version__,
    description="llm judge",
    author="Sergei Porkhun",
    author_email="ser.porkhun41@gmail.com",
    packages=find_packages(),
    install_requires=find_requirements(),
    package_data={
        "llm_judge": [
            "arena_hard/data/arena_hard_en/*.zip",
            "arena_hard/data/arena_hard_en/model_answer/*.zip",
            "arena_hard/data/arena_hard_en/model_judgment/gpt-4-1106-preview/*.zip",
            "arena_hard/data/arena_hard_ru/*.zip",
            "arena_hard/data/arena_hard_ru/model_answer/*.zip",
            "arena_hard/data/arena_hard_ru/model_judgment/gpt-4-1106-preview/*.zip",
            "mt_bench/data/*.zip",
            "mt_bench/data/mt_bench_en/*.zip",
            "mt_bench/data/mt_bench_en/reference_answer/*.zip",
            "mt_bench/data/mt_bench_ru/*.zip",
            "mt_bench/data/mt_bench_ru/reference_answer/*.zip",
            "livebench/data/livebench/coding/*.zip",
            "livebench/data/livebench/data_analysis/*.zip",
            "livebench/data/livebench/instruction_following/*.zip",
            "livebench/data/livebench/language/*.zip",
            "livebench/data/livebench/math/*.zip",
            "livebench/data/livebench/reasoning/*.zip",
        ]
    },
)
