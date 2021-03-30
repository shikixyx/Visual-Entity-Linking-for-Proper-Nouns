# Visual-Entity-Linking-for-Proper-Nouns

## 元論文

<https://drive.google.com/file/d/1hw-6T3U4N9U3qu-xItc0HI7lXJSftZ7k/view?usp=sharing>

## 実験の概要と手順

画像の中のオブジェクトの固有名詞を特定する。そのために、Wikidata上のentityに属する画像と比較し、類似度の高いものをサジェストする。

1. Wikidata上のentityに紐づく画像をダウンロード
2. 取得した画像をベクトル化し、pickleかjoblibで保存
3. 使用するテストデータのIDを決定
4. 各テスト画像と他の画像とのL2ノルムを計算し、距離の小さい上位N個を取得
5. APやTOP-N errorを計算

実験に使用したコードは上記の番号とファイル名の番号が対応している。

## 事前準備

- Wikimediaのテーブルdmpをダウンロード
  - ダウンロード元（日時）：<https://dumps.wikimedia.org/commonswiki/>
  - ダウンロード元（最新）：<https://dumps.wikimedia.org/commonswiki/latest/>
  - 必要なテーブル：
    - page
    - page_props
    - iwilinks
- mysqlサーバーを構築し、上記のテーブルをインポート
  - istクラスタ上に構築する場合は、singularityを使う
  - 参考ページ：<https://github.com/ddbj/singularity_mysql>
  - インポートだけで1週間程度かかるので注意
- インポートしたテーブルから、不要データを取り除いたテーブルを作成(普通にクエリ投げると遅いため)
  - `page_props`テーブルから、`page_props = "wikibase_item"`のもののみを抽出したもの
  - `iwilinks`テーブルから、`iwi_title LIKE 'Q%'`(Qから始まるもの)のみを抽出したもの

```sql
CREATE TABLE page_wikibase_item AS SELECT * FROM page_props WHERE page_props = "wikibase_item";

CREATE TABLE iwilinks_Q as SELECT * FROM iwlinks WHERE iwl_title LIKE 'Q%'
```

## 注意

- common以下のファイルに共通ロジックを格納してますが、01_download_data.pyでは使ってません。
- 01、以降のファイルは、変数名がかなり適当です。
- オブジェクトを一時保存するのに使うライブラリとしてjoblibとpickleが混在しています。これは最初はpickleを使ってましたが、容量が大きくなりすぎたため、途中からjoblibに切り替えたためです。この差異を吸収するために、common/util.pyに出力、保存用の関数を作って、それを利用するようにしてます。が、最初の方のファイルではこうなってないです。
- 正直、01以外のファイルは作り直した方が早い気がします...
- 画像は量が多い（1TB以上）ので、ダウンロードし直すことは勧めます。ist-clusterを使えるなら、`/home/u00237/thesis/images`を使うことを進めます。使えない場合は、data/fileurlsにあるオブジェクトに、各entityに紐づく画像のURL一覧が入っているので、そのURLを元に保存することを勧めます。

## 実験の詳細

### 01.Wikidata上のentityに紐づく画像をダウンロード

- SPARQLで取得したWikidata上のentityの一覧に紐づくWikimediaの画像をダウンロードする。
- ただし、子カテゴリ（例：親カテゴリ＝Shinzo Abe 、子カテゴリ＝Shinzo Abe in 2021　など）はダウンロードできない。

※参考図
![gazou](https://i.imgur.com/veyQGsW.jpg)

- 01_download_data.py にあるパラメータ変数を設定し、実行
- 実験時はpython 3.8.2を使用
- 先にmysqlサーバを上げておく必要がある。ist-clusterでは
  - `/home/u00237/singularity/singularity_mysql/start_container.sh`
  - このコマンドを実行した後に、pythonファイルを実行し、使用が終わったら
  - `singularity instance.stop mysql`
  - でmysqlサーバを停止させる
- 各カテゴリ毎のURLの一覧はdataフォルダ以下のpickleファイルを参照

### 02.取得した画像をベクトル化する

- 01.で取得した画像をPytorchに入っているモデル、及びopencvに含まれるSIFTを使ってベクトル化する。
- 各画像はカテゴリごとに別ディレクトリに含まれているとする(`images/dog`、`images/cat`など)
- ベクトル化に時間がかかるため、並列で実行できるようになっている。並列で実行するには分割数とindex(=分割したうち、何番目か)を指定すれば良い。ロジックとしては、ディレクトリに含まれる画像を、名前でそーとして、分割数個のリストに分けた後、index番目のリストに対してベクトル化を実行する。
- 実行にはpytorchが必要。ist-cluster上で実行するにはanacondaを使うと楽。
- 実行時に必要な引数は、ファイル内に説明が書いてあります。
- slurmで分割して実行した際は、引数を入れた.shファイルを複数作成し、それをsbatchして実行してました。

### 03.テストデータを作成

- 画像ディレクトリ下にあるファイルの一部をテストデータとして記録する。
- それぞれの画像ファイルがテストデータか否かを判定するための辞書{"_filename_": True or False}を作る。

### 04.各テスト画像と他の画像とのL2ノルムを計算し、距離の小さい上位N個を取得

- 03.で記録されたテストデータと、その他の画像との距離を計算する
- 計算はCNN、SIFTのいずれもL2ノルムを使用した
- 愚直にやると実行に1週間程度かかるため、02.と同様に並列で実行できるようにしてある

### 05.APやTOP-N errorを計算

- 04.で生成した、各テスト画像と、それへの上位N個の類似画像のデータを元に、APとTop-N errorを計算

## 実験備忘

- 作業ディレクトリは
  - 元のコード：`/home/u00237/VisualEntityLinking`
  - 画像データ：`/home/u00237/thesis/images`
- mysqlの情報
  - ルートアカウント：`root/hoge`
  - ユーザアカウント：`u00237/u00237`
  - 接続ポート：52004
    - 設定は`/home/u00237/singularity/singularity_mysql/my_mysql.cnf`に記載
  - 起動するには `home/u00237/singularity/singularity_mysql/start_container.sh`を実行
