import sys

sys.path.append(".")

from collections import defaultdict
import os
import numpy as np
import random
from tqdm import tqdm

from common.util import *

### 07. Create testdata (flg)

## カテゴリの指定
CATEGORY = "dog"
if sys.argv and len(sys.argv) >= 2:
    CATEGORY = sys.argv[1]

## 使えるフォーマット
ALLOW_FORMAT = [
    "jpg",
    "jpeg",
    "jpe",
    "jp2",
    "bmp",
    "dib",
    "png",
    "tiff",
    "tif",
    "pbm",
    "pgm",
    "ppm",
    "pxm",
    "pnm",
    "svg",
]

## 画像ディレクトリ
image_dir = image_dir + CATEGORY + "/"
path, dirs, files = next(os.walk(image_dir))

## 準備
## テストデータ1: ランダムに25%のデータを選ぶ
## テストデータ2: 各キーに対して、1つデータを選ぶ
## {"filename": True or False} の辞書を作る

TEST_DATA_1 = {}
TEST_DATA_2 = {}
QIDs = set()
q_files = defaultdict(list)

for filename in tqdm(files):
    ## キーを取得
    qid = filename.split("_")[0][1:]

    ## キーが含まれてないなら追加
    if not qid in QIDs:
        QIDs.add(qid)

    ## 辞書に追加
    TEST_DATA_1[filename] = False
    TEST_DATA_2[filename] = False

    ## キーごとのリストに追加
    q_files[qid].append(filename)


## テストデータ1を作る
num_test_data_1 = len(files) // 4
random.shuffle(files)

for filename in files[:num_test_data_1]:
    TEST_DATA_1[filename] = True

## テストデータ2を作る
for qid in tqdm(list(QIDs)):
    data = q_files[qid]

    if data:
        cnt = 0
        while cnt < 100:
            random.shuffle(data)
            t = data[0]
            format = t.split(".")[-1].lower()

            if format in ALLOW_FORMAT:
                break

            cnt += 1

        TEST_DATA_2[t] = True


## SAVE
test_data_1_pickle_file = "testdata_" + CATEGORY + "_1.pickle"
test_data_2_pickle_file = "testdata_" + CATEGORY + "_2.pickle"

save_object(TEST_DATA_1, test_data_1_pickle_file)
save_object(TEST_DATA_2, test_data_2_pickle_file)

