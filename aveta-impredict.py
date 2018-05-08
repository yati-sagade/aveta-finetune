import os
import sys
import random
import argparse

from keras.models import load_model, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, Input
from matplotlib import pyplot as plt
import numpy as np
from scipy.misc import imread, imresize
from keras.callbacks import ModelCheckpoint, EarlyStopping

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='small dataset run')
    parser.add_argument('model_file', help='path to model file')
    parser.add_argument('imgdir', help='image directory')
    args = parser.parse_args()

    model_file = args.model_file
    imgdir = args.imgdir
    filenames = sorted(filter(lambda t: t.endswith('.jpg'), os.listdir(imgdir)),
                       key=lambda t: int(t.split('.')[0]))
    random.shuffle(filenames)
    if args.test:
        filenames = filenames[:100]
    #model = get_trainable_model(model_file)
    model = load_model(model_file)
    model.compile(optimizer='adam', loss='mean_squared_error')
    #model.save('our_model.h5')
    speedfile = os.path.join(imgdir, 'speeds.txt')
    speeds = {}
    with open(speedfile) as fp:
        for line in fp:
            imfile, _, _, left_speed, right_speed = line.strip().split(',')
            speeds[imfile] = [left_speed, right_speed]
    
    nbtest = int(len(filenames)*0.1)
    test_filenames, train_filenames = filenames[:nbtest], filenames[nbtest:]
    test_x = np.array([
        imresize(imread(os.path.join(imgdir, f)), (120, 160))
        for f in test_filenames
    ])
    train_x = np.array([
        imresize(imread(os.path.join(imgdir, f)), (120, 160))
        for f in train_filenames
    ])
    test_y, train_y = [np.array([speeds[f] for f in fileset])
                        for fileset in (test_filenames, train_filenames)]
    #test_y = np.array([test_y[:,0], test_y[:,1]])
    #train_y = np.array([train_y[:,0], train_y[:,1]])

    print('Shapes: train_x: {}, train_y: {}, test_x: {}, test_y: {}'.format(
        train_x.shape, train_y.shape, test_x.shape, test_y.shape
    ));

    model_dir = os.path.dirname(os.path.abspath(model_file))
    model_file_basename = os.path.basename(model_file)
    best_file_name = os.path.join(model_dir, 'best_'+model_file_basename)
    save_best = ModelCheckpoint(best_file_name,
                                save_best_only=True,
                                verbose=1,
                                mode='min')

    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=5,
                               verbose=0,
                               mode='auto')

    cbs = [save_best, early_stop]

    batch_size = 320 if not args.test else 64
    model.fit(train_x,
              train_y,
              nb_epoch=100,
              validation_split=0.1,
              callbacks=cbs,
              batch_size=batch_size)
    
    test_eval = model.evaluate(test_x, test_y, batch_size=batch_size)
