# Model structure
model = Sequential()
#
model.add(Conv2D(32,(3,3),activation="relu",
                 input_shape=(105,165,3)))
model.add(MaxPool2D((2,2)))
#
model.add(Conv2D(64,(3,3),
                 activation="relu",padding="same"))
model.add(Dropout(0.2))
model.add(MaxPool2D((2,2)))
#
model.add(Conv2D(128,(3,3),
                 activation="relu",padding="same"))
model.add(Dropout(0.5))
model.add(MaxPool2D((2,2)))
#
model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2,activation="sigmoid"))

#Model compilation
model.compile(optimizer="adam" ,loss="binary_crossentropy", metrics=["accuracy"])

#Summary of the model
print(model.summary())

#Model run

model.fit(train_batches, validation_data=valid_batches, epochs=10, verbose=1)