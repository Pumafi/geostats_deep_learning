model = keras.Sequential(
    [
        keras.layers.Dense(100, activation="tanh", name="layer1"),
        keras.layers.Dense(32, activation="softmax", name="output"),
    ]
)
# Call model on a test input
x = tf.ones((1, 784))
y = model(x)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.SGD(learning_rate=.001),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=32, epochs=15, validation_split=0.2)

import pandas as pd

history_df = pd.DataFrame(history.history)
history_df["epoch"] = history.epoch

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(12, 6))
history_df.plot(x="epoch", y=["loss", "val_loss"], ax=ax0)
history_df.plot(x="epoch", y=["accuracy", "val_accuracy"], ax=ax1);