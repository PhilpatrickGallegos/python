import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Model
(X_train, y_train),(X_test, y_test) = mnist.load_data()

#show image
#plt.imshow(x_train[0],cmap='gray')
#plt.show()

#reshape our data from our model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

#one-hot encoding target column Y
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#print(y_test[0])

#build simple sequential model
model = Sequential()
model.add(Conv2D(64, kernel_size=3,name="input", activation='sigmoid',input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3,name="hidden", activation='sigmoid'))
model.add(Flatten())
model.add(Dense(10,name="output",activation='softmax'))

#set up costfunction sgd type and print metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train model
#model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=1)

layer_name = 'hidden'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(X_test)

print (intermediate_output)
#a = (np.asarray(intermediate_output)[2]).T[1] #(10000,24,24,32) -> (24,24,32)->(32,24,24) ->(24,24)
columns = 4
rows = 8
w=20
h=20
original = np.asarray(intermediate_output)
shapes = original.shape

for kk in range(shapes[0]):
  a = (original)[kk].T
  fig=plt.figure(figsize=(8, 8))
  for i in range(1, shapes[3]+1):
    img = np.random.randint(10, size=(h,w))
    fig.add_subplot(rows, columns, i)
    plt.imshow(a[i-1], cmap='gray')
  plt.savefig(str(kk)+"cnn.png")
  plt.close()
