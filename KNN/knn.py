import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

K = 3

#
# KNN algorithm that predicts emotion from a feature [valence, arousal]
#
def predict_emotion(valence, arousal):
    #init model
    model = KNeighborsClassifier(n_neighbors=K)
    
    # read data
    data = pd.read_csv('KNN/train_data.csv')
    features = data.iloc[:,[0,1]].values
    labels = data.iloc[:,2].values
    
    # Train the model using the training sets
    model.fit(features,labels)

    #Predict Output
    prediction = model.predict([[valence,arousal]]) 

    return prediction[0]