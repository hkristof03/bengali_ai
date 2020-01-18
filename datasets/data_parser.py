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

def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        pad_width = ((t,b), (l,r))
    return pad_width

def crop_object(img, thresh=220, maxval=255, square=True):
    """
    Source: https://stackoverflow.com/questions/49577973/how-to-crop-the-biggest-object-in-image-with-python-opencv
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    # threshold to get just the signature (INVERTED)
    retval, thresh_gray = cv2.threshold(
        gray, thresh=thresh, maxval=maxval, type=cv2.THRESH_BINARY_INV
    )
    contours, hierarchy = cv2.findContours(
        thresh_gray,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    # Find object with the biggest bounding box
    mx = (0,0,0,0)      # biggest bounding box so far
    mx_area = 0
    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
        area = w*h
        if area > mx_area:
            mx = x,y,w,h
            mx_area = area
    x,y,w,h = mx

    crop = img[y:y+h, x:x+w]

    if square:
        pad_width = get_pad_width(crop, max(crop.shape))
        crop = np.pad(
            crop, pad_width=pad_width, mode='constant', constant_values=255
        )

    return crop

def save_images(df, path_save, crop=False):
    """
    """
    for index, row in tqdm(df.iterrows()):
        label = df.loc[index].values[0]
        data = df.loc[index].values[1:].reshape(137, 236).astype('uint8')
        if crop:
            data = crop_object(data)
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




if __name__ == '__main__':
    args = parse_args()
    process_parquet_files(args.train, args.nprocess, crop=args.crop)
