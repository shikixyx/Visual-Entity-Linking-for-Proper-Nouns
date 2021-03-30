import sys

sys.path.append(".")

from tqdm import tqdm
import os
import sys
import heapq
import gc

import cv2 as cv
import numpy as np

from common.util import *

"""
引数1: CATEGORY
引数2: NFEATURE
引数3: topN
引数4: 分割数
引数5: index
"""

print_time_with_str(" ".join(sys.argv))

## カテゴリの指定
CATEGORY = "car"
if sys.argv and len(sys.argv) >= 2:
    CATEGORY = sys.argv[1]

## FEATUREの指定
NFEATURE = 128
if sys.argv and len(sys.argv) >= 3:
    NFEATURE = sys.argv[2]

## TOP-N
TOPN = 5
if sys.argv and len(sys.argv) >= 4:
    TOPN = int(sys.argv[3])

## SET divide
DIVIDE = 10
DIVIDE_FLG = True
if sys.argv and len(sys.argv) >= 5:
    DIVIDE = int(sys.argv[4])
    DIVIDE_FLG = True

## SET index
DIVIDE_INDEX = 0
if sys.argv and len(sys.argv) >= 6:
    DIVIDE_INDEX = int(sys.argv[5])

## LOAD: テストデータ
test_data_1_pickle_file = "testdata_" + CATEGORY + "_1.pickle"
test_data_2_pickle_file = "testdata_" + CATEGORY + "_2.pickle"

TEST_DATA_1 = load_object(test_data_1_pickle_file)
TEST_DATA_2 = load_object(test_data_2_pickle_file)


## LOAD: 画像のベクトル
vectors = []

div_num = CATEGORIES[CATEGORY]

for i in tqdm(range(div_num)):
    ## vector_SIFT_128_car_100_97.joblib
    vector_file = "vector_SIFT_{}_{}_{}_{}.joblib".format(
        NFEATURE, CATEGORY, div_num, i
    )
    t = load_object(vector_file)
    vectors += t
    gc.collect()

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


## 似ているキーポイントの数
bf = cv.BFMatcher()

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=10)  # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params, search_params)


def get_match_num(des1, des2):
    global bf
    good = 0

    try:
        matches = bf.knnMatch(des1, des2, k=2)
        # matches = flann.knnMatch(des1, des2, k=2)
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good += 1
    except Exception as e:
        return 0

    return good


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

        dist = get_match_num(v, v_v)

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
topN_1_file = "topN_SIFT_{}_{}_{}_{}_1".format(CATEGORY, NFEATURE, DIVIDE, DIVIDE_INDEX)
topN_2_file = "topN_SIFT_{}_{}_{}_{}_2".format(CATEGORY, NFEATURE, DIVIDE, DIVIDE_INDEX)

save_object(data1_dist, topN_1_file)
gc.collect()

save_object(data2_dist, topN_2_file)
