import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


data = load_breast_cancer()
X = data.data
y = data.target 
print (data.feature_names)
# we will only take two features for visualization purposes
X= data.data[:,[2,4]]# 'mean perimeter' & 'mean smoothness'

# Standardizing the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# add bias (x_0 = 1) term to X
X_bias = np.hstack((X, np.ones((X.shape[0],1))))
print(X_bias)