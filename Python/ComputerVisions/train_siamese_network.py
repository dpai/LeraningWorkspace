from siamesenetworks import build_siamese_model
import config
import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.datasets import mnist
import numpy as np


print("{INFO} loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX/255.0
testX = testX / 255.0

trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

print("[INFO] preparing positive and negative pairs")
(pairTrain, labelTrain) = utils.make_pairs(trainX, trainY)
(pairTest, labelTest) = utils.make_pairs(testX, testY)

print("[INFP] building siamese network...")
imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)
featureExtractor = build_siamese_model(config.IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

distance = Lambda(utils.euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)

print("[INFO] compiling model...")
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("[INFO] training model...")
history = model.fit(
    [pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
    validation_data = ([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
    batch_size = config.BATCH_SIZE,
    epochs=config.EPOCHS
)

print("[INFO] saving siamese model...")
model.save(config.MODEL_PATH)

print("[INFO] saving Siamese model...")
utils.plot_training(history, config.PLOT_PATH)

