# K-Nearest-Neighbours algorithm
some logic for performing KNN on a new valence/arousal pair obtained from predict.py

## train

to train the model you can use the api to label your expressions like so:

`http://localhost:9090/train/<YOUR_LABEL>`

The resulting feature is saved in `KNN/train_data.csv`
