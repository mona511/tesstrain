from pathlib import Path
from typing import Union
from src.training.tesstrain_utils import run_command


def run(lang:str, tessdata_dir:Union[str, Path], output_dir:Union[str, Path]):
    run_command(
        "combine_tessdata",
        "-e",
        f"{tessdata_dir / (f'{lang}.traineddata')}",
        f"{output_dir / (f'{lang}.lstm')}",
    )
