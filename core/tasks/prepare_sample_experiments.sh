#!/bin/bash

# NOTE this is to create multiple experiment specified from sample.json and output to stdio
# Thus we can output it to a file or pipe it to a executor
python training/prepare_experiments.py training/experiments/sample.json
