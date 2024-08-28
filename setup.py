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
    version="0.1",
    description="llm judge",
    author="Sergei Porkhun",
    author_email="ser.porkhun41@gmail.com",
    packages=find_packages(),
    install_requires=find_requirements(),
    entry_points={
        "console_scripts": [
            "clean_judgment=llm_judge.clean_judgment:main",
            "download_pregenerated=llm_judge.download:main",
            "gen_api_answer=llm_judge.gen_api_answer:main",
            "gen_judgment=llm_judge.gen_judgment:main",
            "show_result=llm_judge.show_result:main",
        ],
    },
)
