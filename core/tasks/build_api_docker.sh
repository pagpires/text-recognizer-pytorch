#!/bin/bash

# NOTE: FIRST need conda env export > env_pytorch.yml, then move it to api/env_pytorch.yml
# replaced PyTorch with cpu version, can also remove unneeded packages like jupyter etc

docker build -t text_recognizer_api -f api/Dockerfile .
