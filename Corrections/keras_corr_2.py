output_features=128
kernel_size=(3, 3)
activation='sigmoid'

conv = keras.layers.Conv2D(output_features, kernel_size, activation=activation, use_bias=True)