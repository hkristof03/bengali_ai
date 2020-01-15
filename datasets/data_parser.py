import os
import glob
from multiprocessing import Pool
from functools import partial

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

    df_parquet = pd.DataFrame()

    for file in parquet_files:
        df = pd.read_parquet(file)
        df_parquet = pd.concat([df_parquet, df], axis=0)

    df_split = np.array_split(df_parquet, nprocess)

    def save_images(df, path_save):
        for index, row in tqdm(df.iterrows()):
            label = df.loc[index].values[0]
            data = df.loc[index].values[1:].reshape(137, 236).astype('uint8')
            cv2.imwrite(path_save + f'{label}.jpg', data)

    func = partial(save_images, path_save=ps)
    with Pool(nprocess) as pool:
        pool.map(func, df_split)


if __name__ == '__main__':

    process_parquet_files()
