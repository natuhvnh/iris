import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# Load data
np.random.seed(0) #fix result of np.random
data = pd.read_csv(r"/home/natu/natu/NAL/IRIS/iris.data", sep = ',', header=None)
data = data.values # Shape: 150, 5 
np.random.shuffle(data)
y = data[:, data.shape[1] - 1] # Label - shape: 150, 1
X = data[:, 0 : data.shape[1] - 1].astype(float) # Data - shape: 150, 4
X_train = X[0:105, :] #shape: 120, 4
X_test = X[105:X.shape[0], :] #30, 4
y_train = y[0:105] #shape: 120, 4
y_test = y[105:y.shape[0]] #30, 4
del data, X, y

# Map label => 0, 1, 2 => one hot vector
classes = {'Iris-setosa' : 0,
           'Iris-versicolor' : 1,
           'Iris-virginica' : 2}
y_train = [classes[item] for item in y_train] 
y_test = [classes[item] for item in y_test] 
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
y_train = np_utils.to_categorical(y_train, 3)
y_test = np_utils.to_categorical(y_test, 3)

# softmax
model = Sequential()
model.add(Dense(3, activation = 'softmax'))
Adam = optimizers.Adam(lr=0.1, epsilon=None, decay=0.0, amsgrad=False)
checkpoint = ModelCheckpoint(filepath='weights.hdf5', 
                            verbose=0, 
                            save_best_only= True, 
                            save_weights_only= True,
                            monitor= 'val_acc', 
                            mode = 'max') # save best accuracy model weight
#model.load_weights('weights.hdf5')
model.compile(loss='categorical_crossentropy', optimizer= Adam, metrics=['accuracy'])
model.fit(X_train, y_train, nb_epoch=300, validation_data=(X_test, y_test), callbacks= [checkpoint])
score = model.evaluate(X_test, y_test)
print(score)

# model2 = load_model('weights.hdf5')
# Adam = optimizers.Adam(lr=0.1, epsilon=None, decay=0.0, amsgrad=False)
# model2.compile(loss='categorical_crossentropy', optimizer= Adam, metrics=['accuracy'])
# score = model2.evaluate(X_test, y_test)
# print(score)


