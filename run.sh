#!/bin/bash
module load anaconda/2020.11
export PYTHONUNBUFFERED=1
source activate torch
source activate py37
python trainDGMR.py
