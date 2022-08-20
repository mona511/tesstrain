from pathlib import Path
from typing import Union
from src.training.tesstrain_utils import run_command


def run(
    lang:str,
    new_lang:str,
    tessdata_dir:Union[str, Path],
    output_dir:Union[str, Path],
    checkpoint:str,
):
    output_dir = Path(output_dir)
    
    run_command(
        "lstmtraining",
        "--stop_training",
        f"--continue_from={output_dir / 'exp' / checkpoint}",
        f"--traineddata={output_dir / lang / (lang + '.traineddata')}",
        f"--old_traineddata={tessdata_dir / (lang + '.traineddata')}",
        f"--model_output={output_dir / (new_lang + '.traineddata')}",
    )
