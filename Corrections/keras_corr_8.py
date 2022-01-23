
lenet = keras.Sequential()
lenet.add(layers.Conv2D(filters=6, kernel_size=(5, 5), padding="same", activation='sigmoid', input_shape=(28, 28, 1)))
lenet.add(layers.AveragePooling2D())

lenet.add(layers.Conv2D(filters=16, kernel_size=(5, 5), padding="valid", activation='sigmoid'))
lenet.add(layers.AveragePooling2D())

lenet.add(layers.Flatten())

lenet.add(layers.Dense(units=120, activation='sigmoid'))

lenet.add(layers.Dense(units=84, activation='sigmoid'))

lenet.add(layers.Dense(units=10, activation = 'softmax'))
