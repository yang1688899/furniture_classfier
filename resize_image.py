from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import sys
import urllib3
import multiprocessing

from tqdm import tqdm



def resize_image(img_in_out_path):
    in_path,out_path = img_in_out_path
    if not os.path.exists(out_path):
        img = cv2.imread(in_path)
        img_resize = cv2.resize(img,(224,224))
        cv2.imwrite(out_path, img_resize)

def parse_dir(in_dir,out_dir):
    img_in_out_paths = []
    img_list = os.listdir(in_dir)
    for img_name in img_list:
        in_path = '%s/%s'%(in_dir,img_name)
        out_path = '%s/%s'%(out_dir,img_name)
        img_in_out_paths.append((in_path,out_path))
    return img_in_out_paths


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("error: not enough arguments")
        sys.exit(0)

    in_dir, out_dir = sys.argv[1:]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_in_out_paths = parse_dir(in_dir,out_dir)

    # resize image
    pool = multiprocessing.Pool(processes=12)
    with tqdm(total=len(img_in_out_paths)) as progress_bar:
        for _ in pool.imap_unordered(resize_image, img_in_out_paths):
            progress_bar.update(1)

    sys.exit(1)
