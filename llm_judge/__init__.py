import json
import re
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Union

__version__ = "0.0.3.1"

PART_FILE = re.compile(r"([\w\d]+?)-([\d]+)")


def load_jsonl(path: Union[str, Path]) -> Any:
    data: List[Any] = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def dump_jsonl(data: List[Any], path: Union[str, Path]) -> Any:
    with open(path, "w") as fp:
        for item in data:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_zip(directory: Union[str, Path]) -> None:
    directory = Path(directory)

    def extract(file: Path) -> None:
        with zipfile.ZipFile(file, mode="r") as zip_fp:
            zip_fp.extractall(file.parent)

    with ThreadPoolExecutor(max_workers=16) as pool:
        pool.map(extract, Path(directory).rglob("*.zip"))


extract_zip(Path(__file__).parent)
