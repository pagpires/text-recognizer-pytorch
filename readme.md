### Full Stack DL System (PyTorch)  
----------------------------------
[`text recognizer`](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-project) implementation with PyTorch (all function interfaces remain unchanged)

1. Application structure (adapted from [this](https://fullstackdeeplearning.com/march2019)) 
![project structure](./assets/structure.png)
2. Line detection (left to right: raw image, true label, predicted label)
![detefction](./assets/line_detection.png)
3. Line prediction
![prediction](./assets/line_prediction.png)

4. Accuracy

Line detection:  

| network | Pixel Acc |
| --------|-----------|
| FCN | 0.928 |
| UNet | 0.954 |

Line Prediction:  

| network | Character Acc |
| ------- | ------------- |
| sliding + cnn + rnn +CTC | 0.765 |
| crnn + CTC| 0.808 |
| seq2seq | TBD |


### Usage
-------------
After clone the repo, make a directory `data` at the same level with `core` to store data
```bash
# setup and activate environment
conda env create -f env_pytorch.yml
source activate $(head -1 env_pytorch.yml | cut -d' ' -f2)

# make sure all tests/evaluation passed
cd core
pytest -s text_recognizer/tests/*
pytest -s text_recognizer/evaluation/*

# have fun

# training
sh tasks/train_character_predictor.sh

# loacl prediction
python3 api/app.py

# build, run, and access docker
sh tasks/build_api_docker.sh # build
docker run -p 8000:8000 --name api -it --rm text_recognizer_api # run 
curl "http://0.0.0.0:8000/v1/predict?image_url=http://s3-us-west-2.amazonaws.com/fsdl-public-assets/emnist_lines/or%2Bif%2Bused%2Bthe%2Bresults.png" # access the dockerized API


```


### Dependencies
---------
* conda (instead of pipenv)
* docker
* ubuntu 18.04LTS

### Addition and change log (to original keras-based repo)
-------------
0. Environment
    * Set up conda for managing environment rather than pipenv
1. Single-character prediction
    * DataSequence (wrapper of PyTorch DataLoader)
    * custom `fit`
    * scalar for ylabel instead of one-hot-encoding (processed during DataLoader), thus `NLLLoss` instead of `BCELoss`
2. LineTextRecognizer
    * slide_window, TimeDistributed, ctc_decode
    * implement checkpoint for retrain
    * added a crnn model
3. Tools for exprimentation
4. Experimentation
5. LineDetection
    * rewrite `data_augmentation` for same random transformation on x, y
6. Data versioning
7. Continuous integration
    * support for conda in `config.yml`
8. Deployment
    * loaded LinePredictor and ParagraphTextRecognizer to allow end2end text OCR
    * modified Dockerfile to support conda

### Under development
-------------
1. U-Net for line detection
2. Seq2seq for line prediction
3. Deployment + simple frontend for interaction