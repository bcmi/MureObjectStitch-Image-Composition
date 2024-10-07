import os
import shutil
import cv2
import pandas as pd
import argparse
from tqdm import tqdm

os.chdir('/data')
image_dir = 'dataset/open-images/images'

for split in ['test', 'validation', 'train']:
    download_dir = os.path.join(image_dir, split)
    os.makedirs(download_dir, exist_ok=True)
    
    if 'validation' == split:
        csv_file_path='dataset/open-images/annotations/validation-annotations-bbox.csv'
    elif 'test' == split:
        csv_file_path='dataset/open-images/annotations/test-annotations-bbox.csv'
    else:
        csv_file_path='dataset/open-images/annotations/oidv6-train-annotations-bbox.csv'
    df_val = pd.read_csv(csv_file_path)
    image_ids = set(df_val.ImageID)
    for image_id in tqdm(image_ids):
        current_image_path = os.path.join(image_dir, image_id + '.jpg')
        assert os.path.exists(current_image_path)
        target_image_path  = os.path.join(download_dir, image_id + '.jpg')
        shutil.move(current_image_path, target_image_path)