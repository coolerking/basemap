# 基盤地図情報 ジオイドモデル GeoJSON/Shape 変換ツール

[国土地理院サイト](https://www.gsi.go.jp/) からダウンロード可能な基盤地図情報データを変換して、
プラグインなしの QGIS でも利用可能な形式である Geo JSON/Shape 形式に変換するPythonプログラム。

## 準備

- miniconda を [インストール](https://docs.conda.io/en/latest/miniconda.html#installing) する
- conda-forge 環境を構築する

```shell
conda config --add channels conda-forge
conda config --show channels
conda config --remove channels defaults
conda config --show channels
conda update conda
```

- リポジトリをダウンロードする

```shell
cd hogehoge
git clone https://github.com/exa-internal/basemap.git
```

- conda 環境を構築する

```shell
cd basemap
conda create -n geo --file env_geo.txt
conda activate geo
conda list
cd geoid
```

## 使い方

- [基盤地図情報ダウンロードサイト](https://fgd.gsi.go.jp/download/menu.php) にユーザ登録を行い、ログインする
- 「基盤地図情報 ジオイドモデル」 の 「ファイル選択へ」
- 「ASCII形式のデータのダウンロード」を選択しzipファイルダウンロードする
- ダウンロードしたzipファイルを展開する
- 展開したディレクトリの `program` ディレクトリに格納された `gsigeo2011_ver2_1.asc` (2022/12/01時点)を `download` ディレクトリにコピーする
- 以下のコマンドを入力する

```shell
conda activate geo
python models.py
```

`output` ディレクトリに変換されたファイル群が格納される

