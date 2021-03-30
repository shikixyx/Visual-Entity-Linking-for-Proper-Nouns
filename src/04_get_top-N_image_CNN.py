import sys

sys.path.append(".")

from tqdm import tqdm
import os
import sys
import heapq
import gc

import torch
import torch.nn as nn
from PIL import Image

from common.util import *

"""
引数1: CATEGORY
引数2: MODEL
引数3: LAYER(vgg16のみ)
引数4: topN
引数5: 分割数
引数6: index
"""

print_time_with_str(" ".join(sys.argv))

## カテゴリの指定
CATEGORY = "dog"
if sys.argv and len(sys.argv) >= 2:
    CATEGORY = sys.argv[1]

MODEL_NAME = "resnet18"
if sys.argv and len(sys.argv) >= 3:
    MODEL_NAME = sys.argv[2]

## レイヤーの指定
LAYER = "0"
if sys.argv and len(sys.argv) >= 4:
    LAYER = sys.argv[3]

## TOP-N
TOPN = 10
if sys.argv and len(sys.argv) >= 5:
    TOPN = int(sys.argv[4])

## SET divide
DIVIDE = 20
DIVIDE_FLG = True
if sys.argv and len(sys.argv) >= 6:
    DIVIDE = int(sys.argv[5])
    DIVIDE_FLG = True

## SET index
DIVIDE_INDEX = 0
if sys.argv and len(sys.argv) >= 7:
    DIVIDE_INDEX = int(sys.argv[6])

## use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("USE GPU:", torch.cuda.is_available())
device = torch.device("cpu")
print("use cpu")

## LOAD: テストデータ
test_data_1_pickle_file = "testdata_" + CATEGORY + "_1.pickle"
test_data_2_pickle_file = "testdata_" + CATEGORY + "_2.pickle"

TEST_DATA_1 = load_object(test_data_1_pickle_file)
TEST_DATA_2 = load_object(test_data_2_pickle_file)


## LOAD: 画像のベクトル
vectors = []

## 分割してないやつ
vector_file = "vector_{}_{}_avgpool.joblib".format(MODEL_NAME, CATEGORY)
t = load_object(vector_file)
vectors += t

div_num = CATEGORIES[CATEGORY]

for i in tqdm(range(div_num)):
    vector_file = ""

    if MODEL_NAME == "vgg16":
        # vector_vgg16_uspresident_6_10_9.joblib
        vector_file = "vector_vgg16_{}_{}_{}_{}.joblib".format(
            CATEGORY, LAYER, div_num, i
        )

    elif MODEL_NAME == "resnet18":
        # vector_resnet18_car_avgpool_100_97.joblib
        vector_file = "vector_resnet18_{}_avgpool_{}_{}.joblib".format(
            CATEGORY, div_num, i
        )

    t = load_object(vector_file)
    vectors += t

print("load len : ", len(vectors))
vectors.sort()

target = vectors[:]

## 分割
if DIVIDE_FLG:
    L = len(vectors)
    nums = [(L + i) // DIVIDE for i in range(DIVIDE)]

    start = 0
    for i in range(DIVIDE):
        if i == DIVIDE_INDEX:
            target = vectors[start : start + nums[i]]
            break

        start += nums[i]


## 距離を計算
data1_dist = {}
data2_dist = {}

c = 0
for qid, filename, v in tqdm(target):
    ## テストデータ以外は弾く
    if (not TEST_DATA_1[filename]) and (not TEST_DATA_2[filename]):
        continue

    ## テストデータか?のフラグ
    is_test_data1 = TEST_DATA_1[filename]
    is_test_data2 = TEST_DATA_2[filename]

    ## cosine similarityを算出
    dist_1 = []
    dist_2 = []

    for v_qid, v_filename, v_v in vectors:
        dist = torch.dist(v.to(device).unsqueeze(0), v_v.to(device).unsqueeze(0))
        dist.to("cpu")
        dist = -dist

        ## 1と他
        if is_test_data1 and (not TEST_DATA_1[v_filename]):
            if len(dist_1) > TOPN:
                heapq.heappushpop(dist_1, (dist, v_qid, v_filename))
            else:
                dist_1.append((dist, v_qid, v_filename))
                dist_1.sort()

        ## 2と他
        if is_test_data2 and (not TEST_DATA_2[v_filename]):
            if len(dist_2) > TOPN:
                heapq.heappushpop(dist_2, (dist, v_qid, v_filename))
            else:
                dist_2.append((dist, v_qid, v_filename))
                dist_2.sort()

    ## 比較対象のものはリストへ
    if is_test_data1:
        dist_1.sort(reverse=True)
        data1_dist[filename] = dist_1

    if is_test_data2:
        dist_2.sort(reverse=True)
        data2_dist[filename] = dist_2

    gc.collect()


## SAVE
topN_1_file = ""
topN_2_file = ""

if MODEL_NAME == "resnet18":
    topN_1_file = "top{}_resnet18_{}_avgpool_{}_{}_1".format(
        TOPN, CATEGORY, DIVIDE, DIVIDE_INDEX
    )
    topN_2_file = "top{}_resnet18_{}_avgpool_{}_{}_2".format(
        TOPN, CATEGORY, DIVIDE, DIVIDE_INDEX
    )
elif MODEL_NAME == "vgg16":
    topN_1_file = "top{}_vgg16_{}_{}_{}_{}_1".format(
        TOPN, CATEGORY, LAYER, DIVIDE, DIVIDE_INDEX
    )
    topN_2_file = "top{}_vgg16_{}_{}_{}_{}_2".format(
        TOPN, CATEGORY, LAYER, DIVIDE, DIVIDE_INDEX
    )

save_object(data1_dist, topN_1_file)
gc.collect()

save_object(data2_dist, topN_2_file)
