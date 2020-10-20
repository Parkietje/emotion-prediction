# Emotion Detection From Camera Feed

This repository contains the [code](https://github.com/dkollias/Aff-Wild-models) based on the [paper](https://arxiv.org/pdf/1804.10938.pdf) by D. Kollias, et. al. modified into a predictor (`predict.py`), combined with a face extractor from camera feed (`screenshot.py`) and an http server to provide an API (`server.py`).


## Dependencies

The project uses tensorflow 1.8 and python 3.6.10 (see `requirements.txt`). You should create a virtual python 3.6.10 environment and THEN you can build tensorflow 1.8 from source (see most upvoted answer [here](https://stackoverflow.com/questions/41937915/how-to-pip-install-old-version-of-librarytensorflow))

## Downloading the pretrained model

You can download the pretrained model [here](https://drive.google.com/drive/folders/1yvmRAJT21S33-fNuh6tt8yKdrCp6gHas) and store the unzipped folder in `pretrained_models/`

## Running the scripts

After dependencies are installed you should be able to run each component separately:

### predict.py

Print out the Valence and Arousal values of a given image file

`$ python predict.py path/to/your/image.jpg`

### screenshot.py

Create an jpg file of your face cropped from the camera feed in `screenshots/` directory

`$ python screenshot.py`

### server.py

start a server on port 9090 which you can use to run the predict script on your face whenever you call `http://localhost:9090/screenshot`

`$ python server.py`