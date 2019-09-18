#!/bin/bash
python training/run_experiment.py --save '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "lenet", "train_args": {"batch_size": 1024}}'
