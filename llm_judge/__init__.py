import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

__version__ = "0.0.3"


def extract_zip(directory: Union[str, Path]) -> None:
    directory = Path(directory)

    def extract(file: Path) -> None:
        with zipfile.ZipFile(file, mode="r") as zip_fp:
            zip_fp.extractall(file.parent)

    with ThreadPoolExecutor(max_workers=16) as pool:
        pool.map(extract, Path(directory).rglob("*.zip"))


extract_zip(Path(__file__).parent)
