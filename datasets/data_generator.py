import os

import argparse
import yaml
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

def dump_dict_yaml(yaml_dict, path_yaml):
    with open(path_yaml, 'w') as f:
        yaml.dump(yaml_dict, f)


class DataGenerator(object):
    """
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __init__(
        self, path_train_df, path_test_df, dummies, multilabelstratifiedkfold,
        nfolds, holdout, holdout_size, seed, cutmix, datagen_config,
        train_valid_generator, holdout_generator, test_generator, **kwargs
        ):
        self._path_train_df = self.base_path + path_train_df
        self._path_test_df = self.base_path + path_test_df
        self._dummies = dummies
        self._multilabelstratifiedkfold = multilabelstratifiedkfold
        self._nfolds = nfolds
        self._holdout = holdout
        self._holdout_size = holdout_size
        self._seed = seed
        self._cutmix = cutmix
        self._datagen_config = datagen_config
        self._trvd_config = train_valid_generator
        self._trvd_config['directory'] = (self.base_path
            + self._trvd_config['directory'])
        self._holdout_config = holdout_generator
        self._holdout_config['directory'] = (self.base_path
            + self._holdout_config['directory'])
        self._test_config = test_generator
        self._test_config['directory'] = (self.base_path
            + self._test_config['directory'])
        self._df = None
        self._train_df = None
        self._valid_df = None
        self._holdout_df = None

    def parse_train_dataframe(self):
        """
        """
        self._df = pd.read_csv(self._path_train_df)

        if self._multilabelstratifiedkfold:
            from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
            self._df['id'] = self._df['image_id'].apply(
                lambda x: int(x.split('_')[1])
            )
            cols = ['id', 'grapheme_root', 'vowel_diacritic',
                'consonant_diacritic']
            X, y = self._df[cols].values[:, 0], self._df.values[:, 1:]
            self._df['fold'] = np.nan

            mskf = MultilabelStratifiedKFold(
                n_splits=self._nfolds, shuffle=True, random_state=self._seed
            )
            for i, (_, test_index) in enumerate(mskf.split(X, y)):
                self._df.iloc[test_index, -1] = i

            self._df['fold'] = self._df['fold'].astype('int')
            del self._df['id']

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

    def parse_test_dataframe(self):
        """
        """
        self._df_test = pd.read_csv(self._path_test_df)
        self._df_test['image_id'] = self._df_test['image_id'].apply(
            lambda x: x + '.jpg'
        )

    def get_datagenerators_train(self):
        """
        """
        self.parse_train_dataframe()
        if (self._holdout and not self._multilabelstratifiedkfold):
            self._holdout_df = self._df.sample(
                frac=self._holdout_size, random_state=self._seed
            )
            self._df = self._df.loc[
                (~self._df.index.isin(self._holdout_df.index))
            ]

        if (self._holdout and self._multilabelstratifiedkfold):
            import random
            assert (self._holdout_size >= 1 / self._nfolds)

            if (self._holdout_size == 1 / self._nfolds):
                n_holdout_folds = 1
            if (self._holdout_size > 1/ self._nfolds):
                n_holdout_folds = self._holdout_size / round(1/self._nfolds, 1)

            random.seed(self._seed)
            folds = random.sample(range(self._nfolds), n_holdout_folds)
            condition = self._df['fold'].isin(folds)
            self._holdout_df = self._df.loc[condition]
            self._df = self._df.loc[
                (~self._df.index.isin(self._holdout_df.index))
            ]

        self._train_df, self._valid_df = train_test_split(
            self._df, test_size=0.15
        )
        datagen = ImageDataGenerator(
            **self._datagen_config, rescale=1.0/255.0,
        )
        if self._cutmix:
            from cutmix_keras import CutMixImageDataGenerator

            train_generator1 = datagen.flow_from_dataframe(
                self._train_df, **self._trvd_config
            )
            train_generator2 = datagen.flow_from_dataframe(
                self._train_df, **self._trvd_config
            )
            valid_generator = datagen.flow_from_dataframe(
                self._valid_df, **self._trvd_config
            )
            train_generator = CutMixImageDataGenerator(
                generator1=train_generator1,
                generator2=train_generator2,
                img_size=self._trvd_config['target_size'][0],
                batch_size=self._trvd_config['batch_size']
            )
            return train_generator, valid_generator

        else:
            train_generator = datagen.flow_from_dataframe(
                self._train_df, **self._trvd_config
            )
            valid_generator = datagen.flow_from_dataframe(
                self._valid_df, **self._trvd_config
            )
            return train_generator, valid_generator

    def get_datagenerator_holdout(self):
        """
        """
        if self._holdout:
            datagen = ImageDataGenerator(**self._datagen_config, rescale=1.0/255.0)
            holdout_datagen = datagen.flow_from_dataframe(
                self._holdout_df, **self._holdout_config
            )
            return holdout_datagen

    def get_datagenerator_test(self):
        """
        """
        self.parse_test_dataframe()
        datagen = ImageDataGenerator(
            **self._datagen_config, rescale=1.0/255.0
        )
        test_generator = datagen.flow_from_directory(**self._test_config)

        return test_generator


if __name__ == '__main__':
    args = parse_args()
    path_yaml = DataGenerator.base_path + args.pyaml
    configs = parse_yaml(path_yaml)

    dg = DataGenerator(**configs)
    tg, vg = dg.get_datagenerators_train()
    hg = dg.get_datagenerator_holdout()
    #tg = dg.get_datagenerator_test()   TO be implemented correctly
    print(f'Total train DataFrame shape is: {dg._df.shape}')
    print(f'Holdout DataFrame shape is: {dg._holdout_df.shape}')
    print(f'Train DataFrame shape is: {dg._train_df.shape}')
    print(f'Validation DataFrame shape is: {dg._valid_df.shape}')
