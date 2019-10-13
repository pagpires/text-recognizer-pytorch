# Pull a docker that supports conda
FROM continuumio/miniconda3

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Copy env file in order to cache dependencies
COPY api/env_pytorch.yml ./api/env_pytorch.yml

# Install Python dependencies
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN set -ex && conda env create -f ./api/env_pytorch.yml

RUN apt-get update && \
    apt-get install libgl1-mesa-glx -y

# COPY predictor relevant files, and the meta data files that are required
COPY text_recognizer/ ./text_recognizer
COPY api/ ./api
# keep dir structure
RUN mv ./api/resources/data ../data

# Export env set up to bashrc: https://github.com/ContinuumIO/docker-images/issues/89
# RUN echo "source activate $(head -1 ./api/env_pytorch.yml | cut -d' ' -f2)" > ~/.bashrc
# ENV PATH /opt/conda/envs/$(head -1 ./api/env_pytorch.yml | cut -d' ' -f2)/bin:$PATH

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /repo
# need to replace the default sh executable
ENTRYPOINT [ "/bin/bash", "-c", "source activate py36pytorch && python3 -u /repo/api/app.py"] 
