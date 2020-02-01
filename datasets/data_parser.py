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
    parser.add_argument(
        '--crop', type=str2bool, default=False,
        help='Crop an image or not (default: False)'
    )
    return parser.parse_args()

def crop_image(img, threshold=30, maxval=255, resize_size=256):
    """
    """
    _, thresh = cv2.threshold(img, threshold, maxval, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

    idx = 0
    ls_xmin = []
    ls_ymin = []
    ls_xmax = []
    ls_ymax = []
    for cnt in contours:
        idx += 1
        x,y,w,h = cv2.boundingRect(cnt)
        ls_xmin.append(x)
        ls_ymin.append(y)
        ls_xmax.append(x + w)
        ls_ymax.append(y + h)
    xmin = min(ls_xmin)
    ymin = min(ls_ymin)
    xmax = max(ls_xmax)
    ymax = max(ls_ymax)

    img = img[ymin:ymax,xmin:xmax]
    img = cv2.resize(img, (resize_size, resize_size))

    return img

def save_images(df, path_save, crop=False):
    """
    """
    for index, row in tqdm(df.iterrows()):
        label = df.loc[index].values[0]
        data = df.loc[index].values[1:].reshape(137, 236).astype('uint8')
        if crop:
            data = crop_image(data)
        cv2.imwrite(path_save + f'{label}.jpg', data)

def process_parquet_files(train=True, nprocess=10, crop=False):
    """
    """
    ptd = base_path + '/datasets/train_data/'
    if train:
        if crop:
            ps = base_path + '/datasets/train_images_cropped/'
        else:
            ps = base_path + '/datasets/train_images/'
        parquet_files = glob.glob(ptd + 'train*.parquet')
    else:
        if crop:
            ps = base_path + '/datasets/test_images_cropped/'
        else:
            ps = base_path + '/datasets/test_images/'
        parquet_files = glob.glob(ptd + 'test*.parquet')

    if nprocess > 1:
        df_parquet = pd.DataFrame()
        for file in parquet_files:
            df = pd.read_parquet(file)
            df_parquet = pd.concat([df_parquet, df], axis=0)

        df_split = np.array_split(df_parquet, nprocess)
        func = partial(save_images, path_save=ps, crop=crop)
        with Pool(nprocess) as pool:
            pool.map(func, df_split)
    else:
        for file in parquet_files:
            df_parquet = pd.read_parquet(file)
            save_images(df_parquet, ps)

def create_external_dataframe():
    """
    """
    path_ext_df = (base_path + '/datasets/train_data/external_images.csv')
    path_ext_imgs = (base_path +
        '/datasets/external_images/external_images/*.jpg')

    if not os.path.isfile(path_ext_df):
        files = glob.glob(path_ext_imgs)
        d = {'image_id': files}
        df = pd.DataFrame(d)
        df['image_id'] = df['image_id'].apply(
            lambda x: 'ex_' + x.split('/')[-1]
        )
        df.to_csv(path_ext_df, index=False)
    else:
        print("DataFrame with external images already exists!")




if __name__ == '__main__':
    args = parse_args()
    #process_parquet_files(args.train, args.nprocess, crop=args.crop)
    create_external_dataframe()
