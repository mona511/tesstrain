from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union
from src.training.tesstrain_utils import run_command


def run(fonts_dir:Union[str, Path]):
    with TemporaryDirectory() as temp_dir:
        run_command(
            "text2image",
            "--list_available_fonts",
            f"--fontconfig_tmpdir={temp_dir}",
            f"--fonts_dir={fonts_dir}",
        )
