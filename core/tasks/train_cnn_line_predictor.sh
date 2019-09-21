#!/bin/bash
python training/run_experiment.py '{"dataset": "EmnistLinesDataset", "model": "LineModel", "network": "line_cnn_all_conv", "train_args": {"epochs": 20, "batch_size": 128}}'