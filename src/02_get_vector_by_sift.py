import sys

sys.path.append("/home/u00237/VisualEntityLinking/src")

import datetime
from collections import defaultdict
from tqdm import tqdm
import os
import gc

import cv2 as cv
import numpy as np

from common.util import *

### 05-1. get image vector by using resnet

print_time_with_str(__file__)
print_time_with_str(" ".join(sys.argv))


"""
SET PARAMETER
引数1 : CATEGORY
引数2 : MODEL
引数2 : 分割数
引数3 : index
"""

## SET category
CATEGORY = "dog"
if sys.argv and len(sys.argv) >= 2:
    CATEGORY = sys.argv[1]

## SET model
MODEL_NAME = "SIFT"
if sys.argv and len(sys.argv) >= 3:
    MODEL_NAME = sys.argv[2]

## SET divide
DIVIDE = 10
DIVIDE_FLG = False
if sys.argv and len(sys.argv) >= 4:
    DIVIDE = int(sys.argv[3])
    DIVIDE_FLG = True

## SET index
DIVIDE_INDEX = 0
if sys.argv and len(sys.argv) >= 5:
    DIVIDE_INDEX = int(sys.argv[4])

## SET nfeature
NFEATURE = 128
if sys.argv and len(sys.argv) >= 6:
    NFEATURE = int(sys.argv[5])


"""
LOGIC
"""


"""
Prepare  get_vector function
"""

vectors = []
sift = cv.SIFT_create(nfeatures=NFEATURE)


"""
Prepare images
"""


## 画像ファイルの一覧を取得
image_dir = image_dir + CATEGORY + "/"
path, dirs, files = next(os.walk(image_dir))

## 分割数が指定されている場合は、fileを分割
if DIVIDE_FLG:
    files.sort()
    L = len(files)
    nums = [(L + i) // DIVIDE for i in range(DIVIDE)]

    start = 0
    for i in range(DIVIDE):
        if i == DIVIDE_INDEX:
            files = files[start : start + nums[i]]
            break

        start += nums[i]

"""
Calculate vectors
"""


## ベクトルを計算
cnt = 0
for filename in tqdm(files):
    img = cv.imread(image_dir + filename)

    ## 読み込めたらnp.ndarray、NGならNone
    if not type(img) is np.ndarray:
        continue

    i = filename.split("_")[0][1:]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, des = sift.detectAndCompute(gray, None)
    vectors += [(i, filename, des)]

    gc.collect()

    cnt += 1
    if cnt % 1000 == 0:
        now = datetime.datetime.now()
        print("cnt: ", cnt, now)


"""
Save to objects
"""

## SAVE
vector_file = ""

if DIVIDE_FLG:
    vector_file = "vector_{}_{}_{}_{}_{}".format(
        MODEL_NAME, NFEATURE, CATEGORY, DIVIDE, DIVIDE_INDEX
    )
else:
    vector_file = "vector_{}_{}_{}".format(MODEL_NAME, NFEATURE, CATEGORY)

save_object(vectors, vector_file)


print_time_with_str()
