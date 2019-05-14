import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from keras.utils import to_categorical
from mlxtend.plotting import plot_decision_regions
from mlxtend.classifier import SoftmaxRegression

# Load data
np.random.seed(0) #fix result of np.random
data = pd.read_csv(r"D:\NAL\IRIS\iris.data", sep = ',', header=None)
data = data.values # Shape: 150, 5 
np.random.shuffle(data)
y = data[:, data.shape[1] - 1] # Label - shape: 150, 1
X = data[:, 0 : data.shape[1] - 1] # Data - shape: 150, 4
X_train = X[0:105, :] #shape: 120, 4
X_test = X[105:X.shape[0], :] #30, 4
y_train = y[0:105] #shape: 120, 4
y_test = y[105:y.shape[0]] #30, 4
del data, X, y

# One-hot vector label
encoder = LabelBinarizer()
y_train = encoder.fit_transform(y_train) #shape: 120, 3
y_test = encoder.fit_transform(y_test) #shape: 30, 3
print(y_train)
# Softmax
softmax = SoftmaxRegression(eta=0.01, 
                            epochs=500, 
                            minibatches=1, 
                            random_seed=0,
                            print_progress=3)
"""softmax.fit(X_train, y_train)
plot_decision_regions(X, y, clf=softmax)
plt.title('Softmax Regression - Gradient Descent')
plt.show()"""