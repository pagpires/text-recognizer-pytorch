# Pull a docker that supports conda
FROM continuumio/miniconda3

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Copy only the relevant directories to the working diretory
COPY api/env_pytorch.yml ./api/env_pytorch.yml
# Install Python dependencies
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN set -ex && conda env create -f ./api/env_pytorch.yml

SHELL ["/bin/bash", "-c"]
RUN apt-get update && \
    apt-get install libgl1-mesa-glx -y
# RUN echo "source activate py36pytorch" > ~/.bashrc
# ENV PATH /opt/conda/envs/env/bin:$PATH

COPY text_recognizer/ ./text_recognizer
COPY api/ ./api
# keep dir structure
RUN mv ./api/resources/data ../data

# Pull the environment name out of the environment.yml
# RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
# ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /repo

CMD source activate py36pytorch && python3 -u /repo/api/app.py
# CMD python3 /repo/api/app.py
