import numpy as np
from sklearn import svm
import pandas as pd

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

# Map label => 0, 1, 2
classes = {'Iris-setosa' : 0,
           'Iris-versicolor' : 1,
           'Iris-virginica' : 2}
y_train = [classes[item] for item in y_train] 
y_test = [classes[item] for item in y_test] 

"""
# SVM with linear kernel
C = 1 # svm regularization parameters
SVM = svm.SVC(kernel= 'linear', C = C).fit(X_train, y_train)
accuracy = SVM.score(X_test, y_test)
print(accuracy)
"""

"""
# Linear SVM
C = 1 # svm regularization parameters
lin_svm = svm.LinearSVC(C= C).fit(X_train, y_train)
accuracy = lin_svm.score(X_test, y_test)
print(accuracy)
"""

"""
# SVM with RBF (Radial basis function) kernel
C = 1 # svm regularization parameters
rbf_svm = svm.SVC(C= C, kernel= 'rbf', gamma= 0.06).fit(X_train, y_train)
accuracy = rbf_svm.score(X_test, y_test)
print(accuracy)
"""

"""
# SVM with polynomial RBF
C = 1 # svm regularization parameters
poly_svm = svm.SVC(C= C, kernel= 'poly', degree= 2).fit(X_train, y_train)
accuracy = poly_svm.score(X_test, y_test)
print(accuracy)
"""


# SVC: spupport vector classifier
# Linear kernel: 97.7%
# Linear SVM: 95.5%
# RBF SVM: 95.5%
# Poly SVM: 93.3%


