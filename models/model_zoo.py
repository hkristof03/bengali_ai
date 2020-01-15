import os
import argparse

from keras.models import Model
from keras.layers import (Dense, Conv2D, Flatten, MaxPool2D, GlobalAvgPool2D,
    Dropout, BatchNormalization, Input)
from keras.optimizers import Adam

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate data according to config.'
    )
    parser.add_argument(
        '--pyaml', type=str, default='/experiments/efficientnet_config.yaml',
        help='path to yaml config relative to base dir (project dir)'
    )
    return parser.parse_args()

def parse_yaml(path_yaml):
    with open(path_yaml, 'r') as stream:
        configs = yaml.load(stream.read(), Loader=yaml.Loader)
        return configs


def build_model(base_model, input_shape, metrics, loss, loss_weights, **kwargs):

    if base_model == 'resnet50':
        from keras.applications import  ResNet50
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
    if base_model == 'densenet121':
        from keras.applications import DenseNet121
        base_model = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
    if base_model == 'efficientnetb3':
        from keras_efficientnets import EfficientNetB3
        base_model = EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )

    x_in = Input(shape=input_shape)
    x = Conv2D(3, (3, 3), padding='same')(x_in)
    x = base_model(x)

    x = GlobalAvgPool2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    out_grapheme = Dense(168, activation='softmax', name='root')(x)
    out_vowel = Dense(11, activation='softmax', name='vowel')(x)
    out_consonant = Dense(7, activation='softmax', name='consonant')(x)

    model = Model(inputs=x_in, outputs=[out_grapheme, out_vowel, out_consonant])
    model.compile(
        Adam(lr=0.0001),
        metrics=metrics,
        loss=loss,
        loss_weights=loss_weights
    )

    return model


if __name__ == '__main__':

    args = parse_args()
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_conf = base_path + args.pyaml
    config = parse_yaml(path_conf)

    model = build_model(**config)
