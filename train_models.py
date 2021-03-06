import os

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import (ReduceLROnPlateau, ModelCheckpoint,
    CSVLogger, EarlyStopping)
from tensorflow.keras.metrics import Recall, Precision

from datasets.data_generator import (parse_args, parse_yaml, dump_dict_yaml,
    DataGenerator)
from models.model_zoo import build_model
from noisy_student_trainer import NoisyStudentTrainer


class NeuralNetTrainer(object):
    """
    """
    base_path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, kwargs, noisy_config):
        self._config_all = kwargs   # to store the whole config file and write out with results
        self._preprocess_config = kwargs.get('preprocess')
        self._model_config = kwargs.get('model')
        self._callbacks_config = kwargs.get('callbacks')
        self._train_config = kwargs.get('train')
        self._test_config = kwargs.get('test')
        self._test_code = kwargs.get('test_code')
        self._cutmix = kwargs.get('cutmix')
        self._noisy_student = kwargs.get('noisy_student')
        self._noisy_config = noisy_config
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
        if self._test_code:
            self._train_config['epochs'] = 1
        # if cross_valid: seed??
        self._datagen = DataGenerator(**self._preprocess_config)
        train_gen, valid_gen = self._datagen.get_datagenerators_train()
        step_size_train = train_gen.n / train_gen.batch_size
        step_size_valid = valid_gen.n / valid_gen.batch_size
        self.get_callbacks()

        metrics_d = {
            'root': [Recall(name='recall'), Precision(name='precision')],
            'vowel': [Recall(name='recall'), Precision(name='precision')],
            'consonant': [Recall(name='recall'), Precision(name='precision')]
        }

        model = build_model(**self._model_config, metrics=metrics_d)
        train_history = model.fit(
            train_gen,
            steps_per_epoch=step_size_train,
            validation_data=valid_gen,
            validation_steps=step_size_valid,
            callbacks=self._callbacks,
            **self._train_config
        )

        if self._noisy_student['noisy_student_training']:
            iterations = self._noisy_student['student_iterations']
            ns_datagen = self._datagen.get_datagenerator_noisy_student()
            pseudo_df = self.predict_noisy_student(model, ns_datagen)
            del model

            print("Before NoisyStudentTraining:")
            print(pseudo_df.head(5))

            for i in range(iterations):
                ns_tr_dg, ns_val_dg = self._datagen.get_datagenerator_noisy_student(pseudo_df)
                h_dg = self._datagen.get_datagenerator_holdout()
                ns_trainer = NoisyStudentTrainer(**self._noisy_config)
                ns_trainer.train_noisy_student(ns_tr_dg, ns_val_dg, h_dg)
                # ....
        else:
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

    def predict_noisy_student(self, model, ns_datagen):
        """
        """
        filenames = ns_datagen.filenames
        step_size_ns = ns_datagen.n / ns_datagen.batch_size
        metrics_names = model.metrics_names
        results = model.predict(
            ns_datagen, steps=step_size_ns, verbose=1
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
        selection_threshold = self._noisy_student['selection_threshold']
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

        grapheme_root_dummies = np.array(
            pd.get_dummies(pseudo_df['grapheme_root'])
        )
        vowel_diacritic_dummies = np.array(
            pd.get_dummies(pseudo_df['vowel_diacritic'])
        )
        consonant_diacritic_dummies = np.array(
            pd.get_dummies(pseudo_df['consonant_diacritic'])
        )

        list_grapheme_root_dummies = [i for i in grapheme_root_dummies]
        list_vowel_diacritic_dummies = [i for i in vowel_diacritic_dummies]
        list_consonant_diacritic_dummies = [
            i for i in consonant_diacritic_dummies
        ]

        pseudo_df['grapheme_root'] = list_grapheme_root_dummies
        pseudo_df['vowel_diacritic'] = list_vowel_diacritic_dummies
        pseudo_df['consonant_diacritic'] = list_consonant_diacritic_dummies

        return pseudo_df

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

    nnt_config = configs['NeuralNetTrainer']
    nst_config = configs['NoisyStudentTrainer']

    nnt = NeuralNetTrainer(nnt_config, nst_config)
    nnt.train()
