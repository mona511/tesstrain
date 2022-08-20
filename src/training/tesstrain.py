#!/usr/bin/env python3

# (C) Copyright 2014, Google Inc.
# (C) Copyright 2018, James R Barlow
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This script provides an easy way to execute various phases of training
# Tesseract.  For a detailed description of the phases, see
# https://tesseract-ocr.github.io/tessdoc/Training-Tesseract.html.

import logging
import os
import sys

if (sys.version_info.major < 3) or (sys.version_info.major == 3 and sys.version_info.minor < 6):
    raise Exception("Must be using Python minimum version 3.6!")

sys.path.insert(0, os.path.dirname(__file__))
from tesstrain_utils import (
    parse_flags,
    initialize_fontconfig,
    phase_I_generate_image,
    phase_UP_generate_unicharset,
    phase_E_extract_features,
    make_lstmdata,
    cleanup,
)
import language_specific

log = logging.getLogger()


def setup_logging_console(console_level:int=logging.INFO):
    log.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    console.setFormatter(console_formatter)
    log.addHandler(console)


def setup_logging_logfile(logfile):
    logfile = logging.FileHandler(logfile, encoding='utf-8')
    logfile.setLevel(logging.DEBUG)
    logfile_formatter = logging.Formatter("[%(asctime)s] - %(levelname)s - %(name)s - %(message)s")
    logfile.setFormatter(logfile_formatter)
    log.addHandler(logfile)
    return logfile


def run(console_level:int, *args):
    setup_logging_console(console_level)
    ctx = parse_flags(*args)
    logfile = setup_logging_logfile(ctx.log_file)
    if not ctx.linedata:
        log.error("--linedata_only is required since only LSTM is supported")
        sys.exit(1)

    log.info(f"=== Starting training for language {ctx.lang_code}")
    ctx = language_specific.set_lang_specific_parameters(ctx, ctx.lang_code)

    initialize_fontconfig(ctx)
    phase_I_generate_image(ctx, par_factor=8)
    phase_UP_generate_unicharset(ctx)

    if ctx.linedata:
        phase_E_extract_features(ctx, ["lstm.train"], "lstmf")
        make_lstmdata(ctx)

    log.removeHandler(logfile)
    logfile.close()
    cleanup(ctx)
    log.info("All done!")
