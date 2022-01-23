autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.build([None, x_train.shape[1], x_train.shape[2], 1])
autoencoder.encoder.summary()
autoencoder.decoder.summary()