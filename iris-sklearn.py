import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load data
np.random.seed(0) #fix result of np.random
data = pd.read_csv(r"D:\NAL\IRIS\iris.data", sep = ',', header=None)
data = data.values # Shape: 150, 5 
np.random.shuffle(data)
y = data[:, data.shape[1] - 1] # Label - shape: 150, 1
X = data[:, 0 : data.shape[1] - 1].astype(float) # Data - shape: 150, 4
X_train = X[0:105, :] #shape: 120, 4
X_test = X[105:X.shape[0], :] #30, 4
y_train = y[0:105] #shape: 120, 4
y_test = y[105:y.shape[0]] #30, 4
del data, X, y

# Map label sang 0, 1, 2
classes = {'Iris-setosa' : 0,
           'Iris-versicolor' : 1,
           'Iris-virginica' : 2}
y_train = [classes[item] for item in y_train] 
y_test = [classes[item] for item in y_test] 
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# softmax
softmax = LogisticRegression(random_state=0, 
                            solver='lbfgs',
                            multi_class='multinomial', 
                            max_iter= 700).fit(X_train, y_train)
accuracy = softmax.score(X_test, y_test)
print(accuracy)
