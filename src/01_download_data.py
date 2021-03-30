import requests
import mysql.connector as mysql
import re
from collections import defaultdict
import shutil
from tqdm import tqdm

###############################
# Set Parameter
###############################

## 設定するのは
##   - ターゲットとなるWikidataのentityを取得するSPARQL
##   - mysqlサーバへの接続情報
##   - 画像の保存先ディレクトリ
##   - 画像をダンロードする際のユーザーエージェント


## - ターゲットとなるWikidataのentityを取得するSPARQL
## 例：アメリカの大統領
query = """PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?pid ?label WHERE {
    # position held in President of the United States
    ?pid wdt:P39 wd:Q11696 .

    # instance of man
    ?pid wdt:P31 wd:Q5.

    # get name
    OPTIONAL {
        ?pid rdfs:label ?label filter (lang(?label) = "en") .
    }
}
"""

## - mysqlサーバへの接続情報
server_info = {
    "host": "login000",
    "port": "52004",
    "user": "u00237",
    "passwd": "u00237",
    "database": "mywiki",
}

## - 画像の保存先ディレクトリ
save_image_dir = "/home/u00237/thesis/images/tmp/"

## - 画像をダンロードする際のユーザーエージェント
## 参考：https://meta.wikimedia.org/wiki/User-Agent_policy
ua = "VisualEntityLinkingBot/1.0 (hiroaki49@is.s.u-tokyo.ac.jp) download-image/1.0"
headers = {"User-Agent": ua}


###############################
# 1. entityの一覧を取得
###############################

print("1. entityの一覧を取得")

url = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"
data = requests.get(url, params={"query": query, "format": "json"}).json()

ids = []
labels = []
for item in data["results"]["bindings"]:
    labels.append({"name": item["label"]["value"], "pid": item["pid"]["value"]})

    t = item["pid"]["value"]
    match = re.search("\d*$", t)
    ids.append(match.group(0))


###############################
# 2. 各entityのカテゴリ名を取得
###############################

print("2. 各entityのカテゴリ名を取得")

categorynames = {}
db = mysql.connect(
    host=server_info["host"],
    port=server_info["port"],
    user=server_info["user"],
    passwd=server_info["passwd"],
    database=server_info["database"],
)

cur = db.cursor()

for id in tqdm(ids):
    ## iwilinksテーブルからカテゴリ名の候補を取得
    query = "SELECT t2.page_title FROM iwilinks_Q as t1 ,page as t2 where t1.iwl_title = %s and t1.iwl_from = t2.page_id"
    cur.execute(query, ("Q" + str(id),))

    data = cur.fetchall()
    if data and len(data) > 0:
        categorynames[id] = []
        for d in data:
            categorynames[id].append(d[0])

###############################
# 3. 各カテゴリに属する画像のURL一覧を取得する
###############################

print("3. 各カテゴリに属する画像のURL一覧を取得する")

## 接続先API
fileurls = defaultdict(list)
base_url = "https://commons.wikimedia.org/w/api.php"
base_params = {
    "action": "query",
    "generator": "categorymembers",
    "gcmlimit": 500,
    "gcmtype": "file",
    "prop": "imageinfo",
    "iiprop": "url",
    "format": "json",
}

## URL一覧を取得
for id in tqdm(categorynames.keys()):
    categories = categorynames[id]

    for cat in categories:
        params = base_params
        params["gcmtitle"] = "Category:" + cat

        ## try 5 times
        for _ in range(5):
            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                continue

            json = response.json()
            break

        if not "query" in json or not "pages" in json["query"]:
            continue

        for page_id, value in json["query"]["pages"].items():
            if not "imageinfo" in value:
                continue
            for imageinfo in value["imageinfo"]:
                fileurls[id].append(imageinfo["url"])

        ## ファイルURLを取得できたらbreak
        break

###############################
# 4. 画像をダウンロード
###############################

print("4. 画像をダウンロード")

for id in tqdm(fileurls.keys()):
    c = 0
    urls = fileurls[id]
    fail_cnt = 0

    for url in urls:
        c += 1
        # print("download: ", url)
        # ダウンロードしたファイルの命名は「entity+ファイル番号」
        image_name = "Q" + str(id) + "_" + str(c).zfill(3)
        extension = url.split(".")[-1]
        image_name += "." + extension

        # try download for 5 times
        flg = False
        for _ in range(5):
            res = requests.get(url, stream=True, headers=headers)
            if res.status_code == 200:
                flg = True
                with open(save_image_dir + image_name, "wb") as file:
                    res.raw.decode_content = True
                    shutil.copyfileobj(res.raw, file)

                break
            else:
                pass

