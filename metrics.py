import keras.backend as K
import numpy as np
from sklearn.metrics import recall_score

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')
