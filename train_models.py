import os

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (ReduceLROnPlateau, ModelCheckpoint,
    CSVLogger, EarlyStopping)
from tensorflow.keras.metrics import Recall

from datasets.data_generator import (parse_args, parse_yaml, dump_dict_yaml,
    DataGenerator)
from models.model_zoo import build_model

class NeuralNetTrainer(object):
    """
    """
    base_path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, **kwargs):
        self._config_all = kwargs   # to store the whole config file and write out with results
        self._preprocess_config = kwargs.get('preprocess')
        self._model_config = kwargs.get('model')
        self._callbacks_config = kwargs.get('callbacks')
        self._train_config = kwargs.get('train')
        self._test_config = kwargs.get('test')
        self._test_code = kwargs.get('test_code')
        self._datagen = None
        self._callbacks = None

    def get_callbacks(self):
        """
        """
        bp = NeuralNetTrainer.base_path
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

    def train(self):
        """
        """
        if self._test_code == True:
            self._train_config['epochs'] = 1
        # if cross_valid: seed??
        self._datagen = DataGenerator(**self._preprocess_config)
        train_gen, valid_gen = self._datagen.get_datagenerators_train()
        step_size_train = train_gen.n / train_gen.batch_size
        step_size_valid = valid_gen.n / valid_gen.batch_size
        self.get_callbacks()

        metrics_d = {
            'root': Recall(name='recall'),
            'vowel': Recall(name='recall'),
            'consonant': Recall(name='recall')
        }

        model = build_model(**self._model_config, metrics=metrics_d)
        train_history = model.fit_generator(
            train_gen,
            steps_per_epoch=step_size_train,
            validation_data=valid_gen,
            validation_steps=step_size_valid,
            callbacks=self._callbacks,
            **self._train_config
        )
        self.predict_holdout(model)

    def predict_holdout(self, model):
        """
        """
        holdout_datagen = self._datagen.get_datagenerator_holdout()
        filenames = holdout_datagen.filenames
        step_size_holdout = holdout_datagen.n / holdout_datagen.batch_size
        if self._test_config['tta']:
            predictions = []
            tta_steps = self._test_config['tta_steps']
            for i in tqdm(range(tta_steps)):
                preds = model.predict_generator(
                    test_datagen, steps=step_size_test, verbose=1
                )
                predictions.append(preds)
            # To be continued...
        else:
            metrics_names = model.metrics_names
            print(metrics_names)
            results = model.evaluate(
                holdout_datagen, steps=step_size_holdout
            )
            results = [float(i) for i in results]
            d = dict(zip(metrics_names, results))
            self._config_all['results'] = d
            path_results = (self.base_path + '/datasets/predictions/' +
                self._callbacks_config['experiment_name'] + '.yaml')
            dump_dict_yaml(self._config_all, path_results)

            # Get arrays of predictions for later analysis
            results = model.predict(
                holdout_datagen, steps=step_size_holdout
            )
            root_pred = results[0]]
            vowel_pred = results[1]
            consonant_pred = results[2]

            root_pred = [np.amax(i) for i in root_pred]
            vowel_pred = [np.amax(i) for i in vowel_pred]
            consonant_pred = [np.amax(i) for i in consonant_pred]
            print(root_pred)
            print(vowel_pred)
            print(consonant_pred)

            d = {
                'root_pred': root_pred,
                'vowel_pred': vowel_pred,
                'consonant_pred': consonant_pred
            }
            df_pred = pd.DataFrame.from_dict(d)
            df_pred = pd.concat([holdout_datagen._holdout_df, df_pred], axis=1)
            print(df_pred)


    def predict_test(self):
        """
        tgt_cols = ['grapheme_root','vowel_diacritic','consonant_diacritic']
        row_ids = []
        targets = []
        if self._test_config['tta']
            pass
        else:
            preds = model.predict_generator(
                holdout_datagen, steps=step_size_holdout, verbose=1
            )
            for file in filenames:
                for k in range(3):
                    row_ids.append('Test_' + str(j) + ':' + tgt_cols[k])
                    targets.append(np.argmax(preds[k][j]))
        submit_df = pd.DataFrame(
            {'row_id': row_ids, 'target': targets},
            columns=['row_id', 'target']
        )
        path_save_pred = (NeuralNetTrainer.base_path
            + '/datasets/predictions/'
            + self._callbacks_config['experiment_name']
            + '_predictions.csv')
        submit_df.to_csv(path_save_pred)
        """
        # ..... to be continued



if __name__ == '__main__':
    args = parse_args()
    path_yaml = DataGenerator.base_path + args.pyaml
    configs = parse_yaml(path_yaml)

    nnt = NeuralNetTrainer(**configs)
    nnt.train()
