model = keras.Sequential(
    [
        keras.layers.Dense(100, activation="tanh", name="layer1"),
        keras.layers.Dense(32, activation="softmax", name="output"),
    ]
)

# Call model on a test input
x = tf.ones((1, 784))
y = model(x)