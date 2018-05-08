import os
import sys

from keras.models import load_model, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, Input
from matplotlib import pyplot as plt
import numpy as np
from scipy.misc import imread, imresize

def get_model():
    img_in = Input(shape=(120, 160, 3), name='img_in')

    x = Conv2D(24, (5, 5), name='conv2d_1', activation='relu', strides=(2, 2))(img_in)
    x = Conv2D(32, (5, 5), name='conv2d_2', activation='relu', strides=(2, 2))(x)
    x = Conv2D(64, (5, 5), name='conv2d_3', activation='relu', strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), name='conv2d_4', activation='relu', strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), name='conv2d_5', activation='relu', strides=(1, 1))(x)
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu', name='dense_1')(x)
    x = Dropout(rate=0.1, name='dropout_1')(x)
    x = Dense(50, activation='relu', name='dense_2')(x)
    x = Dropout(rate=0.1, name='dropout_2')(x)

    speeds_out = Dense(2, activation='tanh', name='dense_3')(x)

    model = Model(inputs=[img_in], outputs=[speeds_out])
    return model


def get_trainable_model(model_file):
    model = load_model(model_file)
    model.summary()
    our_model = get_model()
    our_model.summary()
    for layer in our_model.layers[:-1]:
        loaded_layer = model.get_layer(name=layer.name)
        layer.set_weights(loaded_layer.get_weights())
        layer.trainable = False
    return our_model


if __name__ == '__main__':
    model_file = sys.argv[1]
    imgdir = sys.argv[2]
    filenames = sorted(filter(lambda t: t.endswith('.jpg'), os.listdir(imgdir)),
                       key=lambda t: int(t.split('.')[0]))
    model = get_trainable_model(model_file)
    model.save('our_model.h5')
    for filename in filenames:
        im = imread(os.path.join(imgdir, filename))
        im = imresize(im, (120, 160))
        batch = np.expand_dims(im, 0)
        pred = model.predict(batch)
        print(pred)
        break

