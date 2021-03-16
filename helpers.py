import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import cv2

def unzip(path, outPath=None):
    option = f'-d {outPath}' if outPath is not None else ''
    os.system(f'unzip {path} {option}')

def revise_fnames(path: str):
    if (path is None):
        raise ValueError('Please enter path')
    filenames = os.listdir(path)
    print(f'Revising {len(filenames)} file names in f{path}...')
    for fname in tqdm(filenames):
        newName = None
        if len(fname) == 5:
            newName = '000' + fname
        elif len(fname) == 6:
            newName = '00' + fname
        elif len(fname) == 7:
            newName = '0' + fname
        else:
            newName = fname
        # print(f'Changing {fname} to {newName}')
        os.system(
            f'mv {os.path.join(path,  fname)} {os.path.join(path, newName)}')


def mv_to_subfolders(path: str):
    pass

def convert_to_categorical(path: str, outPath: str,  num_classes: int):
    filenames = os.listdir(path)
    print(f'converting {len(filenames)} files to categorical format in {path}...')
    for fname in tqdm(filenames):
        mask = cv2.imread(os.path.join(path, fname), cv2.IMREAD_GRAYSCALE)
        mask = tf.keras.utils.to_categorical(mask, num_classes)
        np.save(os.path.join(outPath, fname[:-4]), mask)


def categorical_to_image(mask):
    return np.argmax(mask, axis=-1)


def revise_labels(path: str):
    if (path is None):
        raise ValueError('Please enter path')
    filenames = os.listdir(path)
    print(f'Revising labels for {len(filenames)} files in {path}...')
    for fname in tqdm(filenames):
        outputStr = None
        with open(os.path.join(path, fname), 'r') as f:
            inputStr = f.read()
            outputStr = inputStr
            conversions = {
                '4': ['4', '5', '6', '7', '8', '9', '10', '11', '12'],
                '5': ['13', '14']
            }
            for label in conversions.keys():
                for item in conversions[label]:
                    outputStr = outputStr.replace(item, label)
        with open(os.path.join(path, fname), 'w') as f:
            # print(f'writing to {fname}')
            f.write(outputStr)


if __name__ == '__main__':
    revise_fnames()
    revise_labels()
