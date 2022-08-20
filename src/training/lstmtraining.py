from pathlib import Path
from typing import Union
from src.training.tesstrain_utils import run_command2


def run(
    lang:str,
    new_lang:str,
    tessdata_dir:Union[str, Path],
    output_dir:Union[str, Path],
    max_iterations:int,
):
    output_dir = Path(output_dir)
    
    model_dir = output_dir / "exp"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    args:list[str] = []
    if max_iterations:
        args.append(f"--max_iterations={max_iterations}")
    
    run_command2(
        "lstmtraining",
        f"--continue_from={output_dir / (lang + '.lstm')}",
        f"--traineddata={output_dir / lang / (lang + '.traineddata')}",
        f"--old_traineddata={tessdata_dir / (lang + '.traineddata')}",
        f"--model_output={model_dir / new_lang}",
        f"--train_listfile={output_dir / (lang + '.training_files.txt')}",
        *args,
    )
