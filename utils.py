import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def calculate_class_weights(y):
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    return dict(enumerate(class_weights))
