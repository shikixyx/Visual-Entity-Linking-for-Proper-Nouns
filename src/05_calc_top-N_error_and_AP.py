import sys

sys.path.append(".")

from tqdm import tqdm
import os

import torch
import torch.nn as nn
import cv2 as cv
import numpy as np

from common.util import *

"""
引数1: モデル
"""

## LOAD:
path, dirs, files = next(os.walk(object_dir))

results = {}


## pickelにあるファイルを走査
for filename in tqdm(files):
    if "top10" in filename:
        continue

    key = ""
    if "resnet" in filename or "vgg16" in filename:
        basename, _ = filename.split(".")
        _, model_name, cat, layer, divide, divide_index, test_num = basename.split("_")

        key = "{}_{}_{}".format(cat, layer, test_num)
    elif "SIFT" in filename:
        ## topN_SIFT_{}_{}_{}_{}_2".format(CATEGORY, NFEATURE, DIVIDE, DIVIDE_INDEX)
        basename, _ = filename.split(".")
        _, model_name, cat, layer, divide, divide_index, test_num = basename.split("_")
        key = "{}_{}_{}".format(cat, layer, test_num)

    if not key in results:
        results[key] = {"total": 0, "top1": 0, "top5": 0, "top10": 0, "AP": 0.0}

    datas = load_object(filename)

    for target_filename, l in datas.items():
        qid = target_filename.split("_")[0][1:]
        ids = [x[1] for x in l]

        results[key]["total"] += 1
        if ids[0] != qid:
            results[key]["top1"] += 1

        if not qid in ids:
            results[key]["top10"] += 1

        ## AP
        precision = []
        hit = 0
        for i, e_id in enumerate(ids):
            if e_id == qid:
                hit += 1
                precision.append(hit / (i + 1.0))

        if precision:
            results[key]["AP"] += np.mean(precision)


for key in results.keys():
    v = results[key]
    print(key, ":", v)
