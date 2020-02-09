import os
import random

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


class CutMixImageDataGenerator():
    def __init__(self, generator1, generator2, img_size, batch_size):
        self.batch_index = 0
        self.samples = generator1.samples
        #self.class_indices = generator1.class_indices
        self.n = generator1.samples
        self.generator1 = generator1
        self.generator2 = generator2
        self.img_size = img_size
        self.batch_size = batch_size

    def reset_index(self):  # Ordering Reset (If Shuffle is True, Shuffle Again)
        self.generator1._set_index_array()
        self.generator2._set_index_array()

    def reset(self):
        self.batch_index = 0
        self.generator1.reset()
        self.generator2.reset()
        self.reset_index()

    def get_steps_per_epoch(self):
        quotient, remainder = divmod(self.samples, self.batch_size)
        return (quotient + 1) if remainder else quotient

    def __len__(self):
        self.get_steps_per_epoch()

    def __next__(self):
        if self.batch_index == 0: self.reset()

        crt_idx = self.batch_index * self.batch_size
        if self.samples > crt_idx + self.batch_size:
            self.batch_index += 1
        else:  # If current index over number of samples
            self.batch_index = 0

        reshape_size = self.batch_size
        last_step_start_idx = (self.get_steps_per_epoch()-1) * self.batch_size
        if crt_idx == last_step_start_idx:
            reshape_size = self.samples - last_step_start_idx

        X_1, y_1 = self.generator1.next()
        X_2, y_2 = self.generator2.next()

        cut_ratio = np.random.beta(a=1, b=1, size=reshape_size)
        cut_ratio = np.clip(cut_ratio, 0.2, 0.8)
        label_ratio = cut_ratio.reshape(reshape_size, 1)
        cut_img = X_2

        X = X_1
        for i in range(reshape_size):
            cut_size = int((self.img_size-1) * cut_ratio[i])
            y1 = random.randint(0, (self.img_size-1) - cut_size)
            x1 = random.randint(0, (self.img_size-1) - cut_size)
            y2 = y1 + cut_size
            x2 = x1 + cut_size
            cut_arr = cut_img[i][y1:y2, x1:x2]
            cutmix_img = X_1[i]
            cutmix_img[y1:y2, x1:x2] = cut_arr
            X[i] = cutmix_img

        y = y_1 * (1 - (label_ratio ** 2)) + y_2 * (label_ratio ** 2)
        return X, y

    def __iter__(self):
        while True:
            yield next(self)


class DataGenerator(object):
    """
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __init__(
        self, path_train_df, path_test_df, dummies, multilabelstratifiedkfold,
        nfolds, holdout, holdout_size, seed, augment_predictions,
        datagen_config, train_valid_generator, holdout_generator,
        test_generator, **kwargs
        ):
        self._path_train_df = self.base_path + path_train_df
        self._path_test_df = self.base_path + path_test_df
        self._dummies = dummies
        self._multilabelstratifiedkfold = multilabelstratifiedkfold
        self._nfolds = nfolds
        self._holdout = holdout
        self._holdout_size = holdout_size
        self._seed = seed
        self._augment_predictions = augment_predictions  # Used not to augment holdout set
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

    @staticmethod
    def get_dummy_targets(df):
            """
            """
            grap_dummies = np.array(pd.get_dummies(df['grapheme_root']))
            vow_dummies = np.array(pd.get_dummies(df['vowel_diacritic']))
            cons_dummies = np.array(pd.get_dummies(df['consonant_diacritic']))

            list_grap = [i for i in grap_dummies]
            list_vow = [i for i in vow_dummies]
            list_cons = [i for i in cons_dummies]

            df['grapheme_root'] = list_grap
            df['vowel_diacritic'] = list_vow
            df['consonant_diacritic'] = list_cons

            return df

    def parse_train_dataframe(self):
        """
        """
        df = pd.read_csv(self._path_train_df)

        if self._multilabelstratifiedkfold:
            from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

            df['id'] = df['image_id'].apply(lambda x: int(x.split('_')[1]))
            cols = ['id', 'grapheme_root', 'vowel_diacritic',
                'consonant_diacritic']
            X, y = df[cols].values[:, 0], df.values[:, 1:]
            df['fold'] = np.nan

            mskf = MultilabelStratifiedKFold(
                n_splits=self._nfolds, shuffle=True, random_state=self._seed
            )
            for i, (_, test_index) in enumerate(mskf.split(X, y)):
                df.iloc[test_index, -1] = i

            df['fold'] = df['fold'].astype('int')
            del df['id']
            df['image_id'] = df['image_id'].apply(lambda x: x + '.jpg')

        if self._dummies:
            df = DataGenerator.get_dummy_targets(df)

        del df['grapheme']
        self._df = df

    def parse_test_dataframe(self):
        """
        """
        df = pd.read_csv(self._path_test_df)
        df['image_id'] = df['image_id'].apply(lambda x: x + '.jpg')
        self._df_test = df

    def get_datagenerators_train(self, test_code=False):
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
            if (self._holdout_size > 1 / self._nfolds):
                n_holdout_folds = int(self._holdout_size / round(1/self._nfolds, 1))

            random.seed(self._seed)
            folds = random.sample(range(self._nfolds), n_holdout_folds)
            condition = self._df['fold'].isin(folds)
            self._holdout_df = self._df.loc[condition]
            self._df = self._df.loc[
                (~self._df.index.isin(self._holdout_df.index))
            ]

        if test_code:
            self._df = self._df.iloc[:8000]

        self._train_df, self._valid_df = train_test_split(
            self._df, test_size=0.1
        )
        datagen = ImageDataGenerator(
            **self._datagen_config, rescale=1.0/255.0,
        )

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
            if self._augment_predictions:
                datagen = ImageDataGenerator(
                    **self._datagen_config, rescale=1.0/255.0
                )
            else:
                datagen = ImageDataGenerator(rescale=1.0/255.0)

            holdout_datagen = datagen.flow_from_dataframe(
                self._holdout_df, **self._holdout_config
            )
            return holdout_datagen

    def get_datagenerator_test(self):
        """
        """
        self.parse_test_dataframe()
        if self._augment_predictions:
            datagen = ImageDataGenerator(
                **self._datagen_config, rescale=1.0/255.0
        )
        else:
            datagen = ImageDataGenerator(rescale=1.0/255.0)

        test_generator = datagen.flow_from_directory(**self._test_config)
        return test_generator


class NoisySudentDataGenerator(object):
    """
    """
    base_path = DataGenerator.base_path

    def __init__(
        self, path_external_df, dummies, datagen_config,
        train_valid_generator, test_generator,
        ):
        self._path_external_df = path_external_df
        self._dummies = dummies
        self._datagen_config = datagen_config
        self._tr_val_config = train_valid_generator
        self._test_config = test_generator
        self._external_df = pd.DataFrame()

    def read_external_df(self):
        """
        """
        self._path_external_df = (self.base_path + self._path_external_df)
        self._external_df = pd.read_csv(self._path_external_df)

    def get_datagenerator_test(self, test_code=False):
        """
        """
        if self._external_df.empty:
            self.read_external_df()
        if test_code:
            self._external_df = self._external_df.iloc[:10000]
        # Teacher never gets noise for predictions!!!
        datagen = ImageDataGenerator(rescale=1.0/255.0)
        nsd = datagen.flow_from_dataframe(
            self._external_df,
            **self._test_config
        )
        return nsd

    def get_datagenerator_train(self, pseudo_df):
        """
        """
        train_df, valid_df = train_test_split(
            pseudo_df, test_size=0.15
        )
        datagen = ImageDataGenerator(
            **self._datagen_config, rescale=1.0/255.0
        )
        ns_tr_datagen = datagen.flow_from_dataframe(
            train_df, **self._tr_val_config
        )
        ns_val_datagen = datagen.flow_from_dataframe(
            valid_df, **self._tr_val_config
        )
        return ns_tr_datagen, ns_val_datagen





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
