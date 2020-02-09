import os

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import (ReduceLROnPlateau, ModelCheckpoint,
    CSVLogger, EarlyStopping)
from tensorflow.keras.metrics import Recall, Precision

from datasets.data_generator import (parse_args, parse_yaml, dump_dict_yaml,
    DataGenerator, NoisySudentDataGenerator)
from models.model_zoo import build_model


class NoisyStudentTrainer(object):
    """
    """
    base_path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, datagen, trainer):
        #self._config_all = kwargs
        self._model_config = trainer.get('model')
        self._callbacks_config = trainer.get('callbacks')
        self._train_config = trainer.get('train')
        self._selection_threshold = trainer.get('selection_threshold')
        self._test_config = trainer.get('test')
        self._test_code = trainer.get('test_code')
        self._datagen = NoisySudentDataGenerator(**datagen)
        self._callbacks = None
        self._df = None
        self._pseudo_df = None

    def get_callbacks(self):
        """
        """
        bp = self.base_path
        exp_name = self._callbacks_config['experiment_name']
        mcp_fp = self._callbacks_config['modelcp']['filepath']
        csv_fn = self._callbacks_config['csvlog']['filename']

        mcp_fp = bp + mcp_fp +  exp_name + '.h5'
        csv_fn = bp + csv_fn + exp_name + '.csv'
        self._callbacks_config['modelcp']['filepath'] = mcp_fp
        self._callbacks_config['csvlog']['filename'] = csv_fn

        reduce_lr = ReduceLROnPlateau(**self._callbacks_config['reducelr'])
        earlystop = EarlyStopping(**self._callbacks_config['earlystop'])
        checkpoint = ModelCheckpoint(**self._callbacks_config['modelcp'])
        csvlog = CSVLogger(**self._callbacks_config['csvlog'])
        self._callbacks = [reduce_lr, earlystop, checkpoint, csvlog]

    def train_noisy_student(self):
        """
        """
        if self._test_code:
            self._train_config['epochs'] = 1

        if not self._callbacks:
            self.get_callbacks()

        ns_tr_datagen, ns_val_datagen =  self._datagen.get_datagenerator_train(
            self._pseudo_df
        )

        metrics_d = {
            'root': [Recall(name='recall'), Precision(name='precision')],
            'vowel': [Recall(name='recall'), Precision(name='precision')],
            'consonant': [Recall(name='recall'), Precision(name='precision')]
        }
        step_size_train = ns_tr_datagen.n / ns_tr_datagen.batch_size
        step_size_valid = ns_val_datagen.n / ns_val_datagen.batch_size
        model = build_model(**self._model_config, metrics=metrics_d)

        train_history = model.fit(
            ns_tr_datagen,
            steps_per_epoch=step_size_train,
            validation_data=ns_val_datagen,
            validation_steps=step_size_valid,
            callbacks=self._callbacks,
            **self._train_config
        )
        ns_test_datagen = self._datagen.get_datagenerator_test()
        self.predict_teacher(model, ns_test_datagen)

        return model


    def predict_teacher(self, model, datagen, df=pd.DataFrame()):
        """
        """
        if not df.empty:
            df.loc[:, 'grapheme_root'] = df['grapheme_root'].apply(
                lambda x: np.argmax(x)
            )
            df.loc[:, 'vowel_diacritic'] = df['vowel_diacritic'].apply(
                lambda x: np.argmax(x)
            )
            df.loc[:, 'consonant_diacritic'] = df['consonant_diacritic'].apply(
                lambda x: np.argmax(x)
            )
            self._df = df  # In order to store the original train + validation data

        df = self._df
        # Itt mindig csak az external_df JÖHET MINT DATAGEN!
        filenames = datagen.filenames
        step_size = datagen.n / datagen.batch_size
        metrics_names = model.metrics_names

        results = model.predict(
            datagen, steps=step_size, verbose=1
        )

        root_pred = results[0]
        vowel_pred = results[1]
        consonant_pred = results[2]
        root_pred = [i for i in root_pred]
        vowel_pred = [i for i in vowel_pred]
        consonant_pred = [i for i in consonant_pred]
        d = {
            'image_id': filenames,
            'grapheme_root': root_pred,
            'vowel_diacritic': vowel_pred,
            'consonant_diacritic': consonant_pred
        }
        pseudo_df = pd.DataFrame.from_dict(d)

        self.select_train_data(pseudo_df, df)


    def select_train_data(self, pseudo_df, df):
        """
        """
        print(f'Original length: {len(pseudo_df)}')

        pseudo_df['gr_max'] = pseudo_df['grapheme_root'].apply(
            lambda x: np.amax(x)
        )
        pseudo_df['vd_max'] = pseudo_df['vowel_diacritic'].apply(
            lambda x: np.amax(x)
        )
        pseudo_df['cd_max'] = pseudo_df['consonant_diacritic'].apply(
            lambda x: np.amax(x)
        )
        # Selection criteria here
        selection_threshold = self._selection_threshold
        condition = (
            (pseudo_df['gr_max'] > selection_threshold)
            #& (pseudo_df['vd_max'] > selection_threshold)
            #& (pseudo_df['cd_max'] > selection_threshold)
        )
        pseudo_df = pseudo_df.loc[condition]
        cols = ['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
        pseudo_df = pseudo_df.loc[:, cols]
        print(f'Selected length: {len(pseudo_df)}')

        pseudo_df.loc[:, 'grapheme_root'] = pseudo_df['grapheme_root'].apply(
            lambda x: np.argmax(x)
        )
        pseudo_df.loc[:, 'vowel_diacritic'] = pseudo_df['vowel_diacritic'].apply(
            lambda x: np.argmax(x)
        )
        pseudo_df.loc[:, 'consonant_diacritic'] = pseudo_df['consonant_diacritic'].apply(
            lambda x: np.argmax(x)
        )

        cols = ['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
        df = df.loc[:, cols]

        pseudo_df = pd.concat([df, pseudo_df], axis=0)
        pseudo_df = DataGenerator.get_dummy_targets(pseudo_df)

        print('_'*50)
        print("Pseudo_df:")
        print(pseudo_df.loc[:, 'grapheme_root'][0].shape)
        print(pseudo_df.loc[:, 'vowel_diacritic'][0].shape)
        print(pseudo_df.loc[:, 'consonant_diacritic'][0].shape)

        self._pseudo_df = pseudo_df
