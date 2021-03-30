import sys

sys.path.append(".")

import datetime
from collections import defaultdict
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

from common.util import *


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
MODEL_NAME = "resnet18"
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


"""
LOGIC
"""

## use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("USE GPU:", torch.cuda.is_available())


"""
Prepare MODEL
- model
- get_vector( img -> torch )
- from_layers
"""

model = None
copy_dest_embedding = {}
vectors = {}
from_layers = {}

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if MODEL_NAME == "resnet18":
    ## LOAD: resnet pretrained model
    model = models.resnet18(pretrained=True)
    model.eval()

    ## 取得するlayerの一覧
    from_layers = {
        "avgpool": 512,
        "layer4": 512,
        "layer3": 256,
        "layer2": 128,
        "layer1": 64,
    }

    ## コピー用の関数とコピー先の変数
    def copy_data(m, i, o):
        copy_dest_embedding[m.lname].copy_(o.data[0, :, 0, 0])

    ## 各レイヤーにコピー関数をfookする
    layers = []
    handles = []
    for name, size in from_layers.items():
        l = model._modules.get(name)

        ## copy先を指定するため、attrを追加
        setattr(l, "lname", name)

        ## copy関数を登録
        handles += [l.register_forward_hook(copy_data)]

        ## 出力先を用意
        vectors[name] = []
        copy_dest_embedding[name] = torch.zeros(size)

    ## ベクトル取得用の関数
    ## copy_dest_embeddingに結果を格納
    def get_vector(img):
        ## Create a PyTorch Variable with the transformed image
        t_img = transform(img)
        t_img = torch.unsqueeze(t_img, 0)

        ## move the input to GPU for speed if available
        if torch.cuda.is_available():
            t_img = t_img.to(device)

        ## Run the model on our transformed image
        model(t_img)

        return


elif MODEL_NAME == "vgg16":
    ## LOAD: resnet pretrained model
    model = models.vgg16(pretrained=True)
    model.eval()

    ## 取得するlayerの一覧
    from_layers = {
        "0": 4096,
        "3": 4096,
        "6": 1000,
    }

    ## コピー用の関数とコピー先の変数
    def copy_data(m, i, o):
        copy_dest_embedding[m.lname].copy_(o.data[0, :])

    ## 各レイヤーにコピー関数をfookする
    layers = []
    handles = []
    for name, size in from_layers.items():
        l = model.classifier[int(name)]

        ## copy先を指定するため、attrを追加
        setattr(l, "lname", name)

        ## copy関数を登録
        handles += [l.register_forward_hook(copy_data)]

        ## 出力先を用意
        vectors[name] = []
        copy_dest_embedding[name] = torch.zeros(size)

    ## ベクトル取得用の関数
    ## copy_dest_embeddingに結果を格納
    def get_vector(img):
        ## Create a PyTorch Variable with the transformed image
        t_img = transform(img)
        t_img = torch.unsqueeze(t_img, 0)

        ## move the input to GPU for speed if available
        if torch.cuda.is_available():
            t_img = t_img.to(device)

        ## Run the model on our transformed image
        model(t_img)

        return


# move the model to GPU for speed if available
if torch.cuda.is_available():
    model.to(device)

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
    i = filename.split("_")[0][1:]

    try:
        img = Image.open(image_dir + filename)
        img = img.convert("RGB")
    except:
        print(filename)
        continue

    get_vector(img)

    for l_name, size in from_layers.items():
        t = torch.zeros(size)
        t.copy_(copy_dest_embedding[l_name]).to("cpu")

        vectors[l_name] += [(i, filename, t)]

    cnt += 1
    if cnt % 1000 == 0:
        now = datetime.datetime.now()
        print("cnt: ", cnt, now)


## remove handles
for h in handles:
    h.remove()


"""
Save to objects
"""

## SAVE
for l_name in from_layers.keys():
    vector_file = ""

    if DIVIDE_FLG:
        vector_file = "vector_{}_{}_{}_{}_{}".format(
            MODEL_NAME, CATEGORY, l_name, DIVIDE, DIVIDE_INDEX
        )
    else:
        vector_file = "vector_{}_{}_{}".format(MODEL_NAME, CATEGORY, l_name)

    save_object(vectors[l_name], vector_file)


print_time_with_str()
