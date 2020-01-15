import os

import argparse
import yaml
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate data according to config.'
    )
    parser.add_argument(
        '--pyaml', type=str, default='/experiments/data_loader.yaml',
        help='path to yaml config relative to base dir (project dir)'
    )
    return parser.parse_args()

def parse_yaml(path_yaml):
    with open(path_yaml, 'r') as stream:
        configs = yaml.load(stream.read(), Loader=yaml.Loader)
        return configs


class DataGenerator(object):
    """
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __init__(self, **kwargs):
        self._path_train_df = DataGenerator.base_path + kwargs.get('path_train_df')
        self._dummies = kwargs.get('dummies')
        self._datagen_config = kwargs.get('datagen_config')
        self._trvd_config = kwargs.get('train_valid_generator')
        self._df = None
        self._train_df = None
        self._valid_df = None

    def parse_train_dataframe(self):
        """
        """
        self._df = pd.read_csv(self._path_train_df)
        self._df['image_id'] = self._df['image_id'].apply(
            lambda x: x + '.jpg'
        )
        if self._dummies == True:
            grapheme_root_dummies = np.array(
                pd.get_dummies(self._df['grapheme_root'])
            )
            vowel_diacritic_dummies = np.array(
                pd.get_dummies(self._df['vowel_diacritic'])
            )
            consonant_diacritic_dummies = np.array(
                pd.get_dummies(self._df['consonant_diacritic'])
            )

            list_grapheme_root_dummies = [i for i in grapheme_root_dummies]
            list_vowel_diacritic_dummies = [i for i in vowel_diacritic_dummies]
            list_consonant_diacritic_dummies = [
                i for i in consonant_diacritic_dummies
            ]

            self._df['grapheme_root_dummies'] = list_grapheme_root_dummies
            self._df['vowel_diacritic_dummies'] = list_vowel_diacritic_dummies
            self._df['consonant_diacritic_dummies'] = list_consonant_diacritic_dummies

            del self._df['grapheme_root']
            del self._df['vowel_diacritic']
            del self._df['consonant_diacritic']
            del self._df['grapheme']

            self._df = self._df.rename(columns={
                'grapheme_root_dummies': 'grapheme_root',
                'vowel_diacritic_dummies': 'vowel_diacritic',
                'consonant_diacritic_dummies': 'consonant_diacritic'
            })
        else:
            del self._df['grapheme']

    def get_datagenerators(self):
        """
        """
        self.parse_train_dataframe()
        self._train_df, self._valid_df = train_test_split(
            self._df, test_size=0.15
        )
        datagen = ImageDataGenerator(
            **self._datagen_config, rescale=1.0/255.0,
        )
        train_generator = datagen.flow_from_dataframe(
            self._train_df, target_size=(137, 236), **self._trvd_config
        )
        valid_generator = datagen.flow_from_dataframe(
            self._valid_df, target_size=(137, 236), **self._trvd_config
        )
        return train_generator, valid_generator


if __name__ == '__main__':
    args = parse_args()
    path_yaml = DataGenerator.base_path + args.pyaml
    configs = parse_yaml(path_yaml)

    dg = DataGenerator(**configs)
    tg, vg = dg.get_datagenerators()
    print(dg._df)