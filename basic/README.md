# 基盤地図情報 基本情報 GeoJSON/Shape 変換ツール

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
cd basic
```

## 使い方

- [基盤地図情報ダウンロードサイト](https://fgd.gsi.go.jp/download/menu.php) にユーザ登録を行い、ログインする
- 「基盤地図情報 基本情報」 の 「ファイル選択へ」
- 必要なメッシュ/都道府県の基本情報を選択しダウンロードする
- ダウンロードしたzipファイルを展開する
- 展開したディレクトリ(FG-GML-..で開始するディレクトリ名)を `basemap/basic/download` ディレクトリ以下にコピー
- 以下のコマンドを入力する

```shell
conda activate geo
cd hogehoge
cd basename/basic
python models.py
```

`output` ディレクトリに変換されたファイル群が格納される

