import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import pickle
import time
save_path = 'NeuronNets'


NAME = 'finger-regognizer-Conv2D-128-6-{}'.format(int(time.time()))
tBoard = TensorBoard(log_dir=f'NeuronNets/{NAME}')

print('Loading dataset(Images)...')
X = pickle.load(open(r'DataSet\Images.dataset', 'rb'))
print('Loading dataset(Lables)...')
y = pickle.load(open(r'DataSet\Lables.dataset', 'rb'))
y = np.array(y, dtype=np.float)
X = np.array(X).reshape(-1, 128, 128)
print(X.shape)

model = Sequential()
model.add(layers.Flatten(input_shape=X.shape[1:]))
model.add(layers.Dense(300, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(X, y, batch_size=100, epochs=5, validation_split=0.1, callbacks=[tBoard])
model.save('input-300-6.h5')
