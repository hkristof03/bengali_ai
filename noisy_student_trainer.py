import os

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import (ReduceLROnPlateau, ModelCheckpoint,
    CSVLogger, EarlyStopping)
from tensorflow.keras.metrics import Recall, Precision

from datasets.data_generator import (parse_args, parse_yaml, dump_dict_yaml,
    DataGenerator)
from models.model_zoo import build_model


class NoisyStudentTrainer(object):
    """
    """
    base_path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, **kwargs):
        self._config_all = kwargs
        self._model_config = kwargs.get('model')
        self._callbacks_config = kwargs.get('callbacks')
        self._train_config = kwargs.get('train')
        self._test_config = kwargs.get('test')
        self._test_code = kwargs.get('test_code')
        self._callbacks = None

    def get_callbacks(self):
        """
        """
        bp = NoisyStudentTrainer.base_path
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

    def train_noisy_student(
        self, ns_tr_datagen, ns_val_datagen, holdout_datagen=None
        ):
        """
        """
        if self._test_code:
            self._train_config['epochs'] = 1

        metrics_d = {
            'root': [Recall(name='recall'), Precision(name='precision')],
            'vowel': [Recall(name='recall'), Precision(name='precision')],
            'consonant': [Recall(name='recall'), Precision(name='precision')]
        }
        self.get_callbacks()
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
        if holdout_datagen:
            self.predict_holdout(model, holdout_datagen)


    def predict_holdout(self, model, holdout_datagen):
        """
        """
        filenames = holdout_datagen.filenames
        step_size_holdout = holdout_datagen.n / holdout_datagen.batch_size

        metrics_names = model.metrics_names
        results = model.evaluate(
            holdout_datagen, steps=step_size_holdout
        )
        results = [float(i) for i in results]
        d = dict(zip(metrics_names, results))
        scores = [
            d['root_recall'], d['vowel_recall'], d['consonant_recall']
        ]
        hma_recall = float(np.average(scores, weights=[2,1,1]))
        d['hier_macro_avg_recall'] = hma_recall
        self._config_all['results'] = d
        path_results = (self.base_path
            + self._test_config['path_save_config']
            + self._callbacks_config['experiment_name'] + '.yaml')
        dump_dict_yaml(self._config_all, path_results)

        # Get arrays of predictions for later analysis
        results = model.predict(
            holdout_datagen, steps=step_size_holdout, verbose=1
        )
        root_pred = results[0]
        vowel_pred = results[1]
        consonant_pred = results[2]

        root_pred = np.argmax(i, axis=1)
        vowel_pred = np.argmax(i, axis=1)
        consonant_pred = np.argmax(i, axis=1)

        d = {
            'image_id': filenames,
            'root_pred': root_pred,
            'vowel_pred': vowel_pred,
            'consonant_pred': consonant_pred
        }
        df_pred = pd.DataFrame.from_dict(d)
        df_pred = df_pred.merge(
            self._datagen._holdout_df,
            how='left',
            on='image_id'
        )
        cols = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
        for col in cols:
            df_pred[col] = df_pred[col].apply(lambda x: np.argmax(x))

        path_predictions = (self.base_path
            + self._test_config['path_predictions']
            + self._callbacks_config['experiment_name'] + '.csv')
        df_pred.to_csv(path_predictions, index=False)

        ## Returns holdout df here in order to train a last model
