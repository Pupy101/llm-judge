import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from pathlib import Path
from typing import List, Union

__version__ = "0.0.3"


class Chdir(AbstractContextManager):
    def __init__(self, path: Union[str, Path]):
        self.path = path
        self._old_cwd: List[str] = []

    def __enter__(self):
        self._old_cwd.append(os.getcwd())
        os.chdir(self.path)

    def __exit__(self, *excinfo):
        path = self._old_cwd.pop()
        if not os.path.exists(path):
            os.makedirs(path)
        os.chdir(path)


def extract_zip(directory: Union[str, Path]) -> None:
    directory = Path(directory)

    def extract(file: Path) -> None:
        with Chdir(file.parent), zipfile.ZipFile(file, mode="r") as zip_fp:
            zip_fp.extractall(directory)

    with ThreadPoolExecutor(max_workers=16) as pool:
        pool.map(extract, Path(directory).rglob("*.zip"))


extract_zip(Path(__file__).parent)
