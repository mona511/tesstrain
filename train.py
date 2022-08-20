import logging
import os
from pathlib import Path
import shutil
import sys
import argparse
import hydra
from tempfile import TemporaryDirectory
from omegaconf import DictConfig
from src.training.tesstrain import (
    setup_logging_console,
    setup_logging_logfile,
    log,
)
from src.training import (
    list_available_fonts,
    tesstrain,
    combine_tessdata,
    lstmtraining,
    composite,
)


class Config(DictConfig):
    stage:int
    console_level:int
    
    fonts_dir:str
    
    lang:str
    fonts:list[str]
    langdata_dir:str
    tessdata_dir:str
    training_text:str
    output_dir:str
    ptsize:int
    save_box_tiff:bool
    tesstrain_args:list[str]
    
    new_lang:str
    max_iterations:int
    
    checkpoint:str


def initialize_config(config:Config):
    file_dir = Path(__file__).resolve().parent
        
    if config.fonts_dir:
        config.fonts_dir = Path(config.fonts_dir)
    else:
        config.fonts_dir = file_dir / "fonts"
    
    if not config.lang:
        config.lang = "jpn"
    
    if config.langdata_dir:
        config.langdata_dir = Path(config.langdata_dir)
    else:
        config.langdata_dir = file_dir / "langdata"
    
    if config.tessdata_dir:
        config.tessdata_dir = Path(config.tessdata_dir)
    else:
        config.tessdata_dir = file_dir / "tessdata_best"
    
    if config.training_text:
        config.training_text = Path(config.training_text)
    else:
        config.training_text = file_dir / "langdata" / config.lang / (config.lang + ".training_text")
    
    if config.fonts and not isinstance(config.fonts, list):
        config.fonts = [config.fonts]
    
    if config.output_dir:
        config.output_dir = Path(config.output_dir)
        config.output_dir.mkdir(parents=True, exist_ok=True)
    
    if config.console_level in ["debug", "DEBUG"]:
        config.console_level = logging.DEBUG
    else:
        config.console_level = logging.INFO
    
    if not config.new_lang:
        config.new_lang = f"new_{config.lang}"
    
    if not config.checkpoint:
        config.checkpoint = f"{config.new_lang}_checkpoint"
    
    if not config.ptsize:
        config.ptsize = 12
        
    if not config.save_box_tiff:
        config.save_box_tiff = False


def initialize_logging(console_level:int=logging.INFO, log_filename:str="tesstrain", output_dir:str=""):
    def _initialize_logging(func):
        def wrapper(*args, **kwargs):
            with TemporaryDirectory() as temp_dir:
                setup_logging_console(console_level=console_level)
                log_file = os.path.join(temp_dir, f"{log_filename}.log")
                logfile = setup_logging_logfile(log_file)
                func(*args, **kwargs)
                log.removeHandler(logfile)
                logfile.close()
                if isinstance(output_dir, str) and output_dir != "" or \
                    isinstance(output_dir, Path) and output_dir is not None:
                    shutil.copy(f"{log_file}", f"{output_dir}")
        return wrapper
    return _initialize_logging


def get_tesstrain_args(config:Config) -> list[str]:
    args:list[str] = []
    if config.lang:
        args.append(f"--lang={config.lang}")
    if config.fonts:
        args.append("--fontlist")
        for font in config.fonts:
            args.append(font)
    if config.fonts_dir:
        args.append(f"--fonts_dir={config.fonts_dir}")
    if config.langdata_dir:
        args.append(f"--langdata_dir={config.langdata_dir}")
    if config.tessdata_dir:
        args.append(f"--tessdata_dir={config.tessdata_dir}")
    if config.training_text:
        args.append(f"--training_text={config.training_text}")
    if config.output_dir:
        args.append(f"--output_dir={config.output_dir}")
    if config.tesstrain_args:
        for arg in config.tesstrain_args:
            args.append(arg)
    args.append("--linedata_only")
    args.append("--noextract_font_properties")
    return args


def my_app(config:Config):
    initialize_config(config)
    
    if config.stage == 0:
        @initialize_logging(logging.DEBUG, "list_available_fonts", config.output_dir)
        def run(config:Config):
            list_available_fonts.run(config.fonts_dir)
        run(config)
    
    if config.stage == 1:
        tesstrain.run(config.console_level, *get_tesstrain_args(config))
    
    if config.stage == 2:
        @initialize_logging(logging.DEBUG, "combine_tessdata", config.output_dir)
        def run(config:Config):
            combine_tessdata.run(
                config.lang,
                config.tessdata_dir,
                config.output_dir,
            )
        run(config)

    if config.stage == 3:
        @initialize_logging(logging.DEBUG, "lstmtraining", config.output_dir)
        def run(config:Config):
            lstmtraining.run(
                config.lang,
                config.new_lang,
                config.tessdata_dir,
                config.output_dir,
                config.max_iterations,
            )
        run(config)

    if config.stage == 4:
        @initialize_logging(logging.DEBUG, "composite", config.output_dir)
        def run(config:Config):
            composite.run(
                config.lang,
                config.new_lang,
                config.tessdata_dir,
                config.output_dir,
                config.checkpoint,
            )
        run(config)


if __name__ == "__main__":
    class Options:
        stage:int
        config:str
        
        def overrides(self) -> list[str]:
            args:list[str] = []
            if self.stage:
                args.append(f"stage={self.stage}")
            return args
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        type=int,
        choices=[0,1,2],
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    option = Options()
    parser.parse_args(args=sys.argv[1:], namespace=option)
    
    with hydra.initialize_config_dir(config_dir=f"{Path(option.config).parent}", version_base=None):
        my_app(hydra.compose(config_name=Path(option.config).name, overrides=option.overrides()))
