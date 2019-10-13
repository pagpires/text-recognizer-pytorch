#!/bin/bash

# pipenv lock --requirements --keep-outdated > api/requirements.txt
# NOTE: need conda env export > env_pytorch.yml first

# sed -i 's/-gpu//g' api/requirements.txt
docker build -t text_recognizer_api -f api/Dockerfile .
