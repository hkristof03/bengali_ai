import os
import glob
from multiprocessing import Pool
from functools import partial
import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Determine train or test parsing and number of processes.'
    )
    parser.add_argument(
        '--train', type=str2bool, default=True,
        help='Train or test parsing (default: Train).'
    )
    parser.add_argument(
        '--nprocess', type=int, default=10,
        help='Number of processes to use (default: 10).'
    )
    return parser.parse_args()

def save_images(df, path_save):
    """
    """
    for index, row in tqdm(df.iterrows()):
        label = df.loc[index].values[0]
        data = df.loc[index].values[1:].reshape(137, 236).astype('uint8')
        cv2.imwrite(path_save + f'{label}.jpg', data)

def process_parquet_files(train=True, nprocess=10):
    """
    """
    ptd = base_path + '/datasets/train_data/'
    if train:
        ps = base_path + '/datasets/train_images/'
        parquet_files = glob.glob(ptd + 'train*.parquet')
    else:
        ps = base_path + '/datasets/test_images/'
        parquet_files = glob.glob(ptd + 'test*.parquet')

    if nprocess > 1:
        df_parquet = pd.DataFrame()
        for file in parquet_files:
            df = pd.read_parquet(file)
            df_parquet = pd.concat([df_parquet, df], axis=0)

        df_split = np.array_split(df_parquet, nprocess)
        func = partial(save_images, path_save=ps)
        with Pool(nprocess) as pool:
            pool.map(func, df_split)
    else:
        for file in parquet_files:
            df_parquet = pd.read_parquet(file)
            save_images(df_parquet, ps)


if __name__ == '__main__':
    args = parse_args()
    process_parquet_files(args.train, args.nprocess)
