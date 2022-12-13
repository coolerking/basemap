# -*- coding: utf-8 -*-
"""
基盤地図情報基本情報管理クラスモジュール。

注意：
使用する前に、基盤地図情報基本情報をダウンロードしておく必要があります。
https://fgd.gsi.go.jp/download/geoid.php より
XML形式のファイルをダウンロード・展開して、
コンストラクタ引数にパスを指定してください。
"""
import os
import gc
import glob
import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt

class BasicModel:
    """
    国土地理院 基盤地図情報 基本情報をあらわすクラス。
    1XMLファイル1インスタンスであらわす。
    """

    def __init__(self, path:str, debug:bool=False) -> None:
        """
        基盤地図情報基本情報XMLファイルを読み込む。

        Parameters
        ----
        path:str
            基盤地図情報基本情報XMLファイルパス
        debug:bool=False
            デバッグオプション
        """
        self.debug = debug
        self.path = path
        self.gdf = self.get_gpd()
        if self.debug:
            self.show_meta()

    def get_gpd(self) -> gpd.GeoDataFrame:
        """
        ジオイドモデルをGeoDataFrame オブジェクトとして取得する。
        座標系はXMLに従う。

        Returns
        ----
        gpd.GeoDataFrame
            ジオイドモデル(ジオイド高:'height'属性)
        """
        gdf = gpd.read_file(self.path)
        if gdf.crs is None:
            print('crs None!')
            raise ValueError(f'crs {gdf.crs} in XML {self.path} is None')
        return gdf

    def show_meta(self):
        """
        メタ情報を表示する。
        """
        print(f'path:    {self.path}')
        print(f'crs:     {self.gdf.crs}')
        print(f'columns: {self.gdf.columns}')
        print(f'length:  {len(self.gdf)}')

    def get_histgram(self, path:str=None):
        """
        ジオイド高値のヒストグラムを表示・保存する。
        負値(値なし相当値-9999.0含む)は対象外とする。

        Parameters
        ----
        path:str
            保存先ファイルパス、指定しない場合表示される。
        """
        # z値（ジオイド高）のみ取得
        (_, _, z) = self.convert_xyz()

        # フォントファミリ指定
        #plt.rcParams['font.family'] = 'Meiryo'

        # figure の作成
        fig = plt.figure(figsize=(8, 6))

        # subplot の追加
        ax = fig.add_subplot()

        # タイトルの作成
        ax.set_title(self.path, size=10)
        ax.hist(z, range(0, int(np.max(np.array(z)))+1))
        
        # 保存先パスが定義されていない場合
        if path is None:
            # グラフを表示
            plt.show()
        else:
            plt.savefig(path)
            if self.debug:
                print(f'saved histgram to {path}')

    def get_scatter2d(self, path:str=None) -> None:
        """
        日本のジオイドデータを2次元散布図に変換する。

        Parameters
        ----
        path:str
            2次元散布図保存先ファイルパス、指定なしの場合は表示
        """
        # subplot の追加
        ax = self.gdf.plot(figsize=(8, 6))

        # タイトルの作成
        ax.set_title(os.path.basename(path), size=10)
 
        # 軸ラベルのサイズと色を設定
        ax.set_xlabel('latitude(degrees)',size=10,color='black')
        ax.set_ylabel('longitude(degrees)',size=10,color='black')

        # 保存先指定なし
        if path is None:
            # 散布図を表示
            plt.show()
        else:
            # 散布図を保存
            plt.savefig(path)
            if self.debug:
                print(f'saved 2d scatter to {path}')

    def save_csv(self, path:str=None) -> None:
        """
        CSV形式ファイルとして保存する。
        緯度、経度、標高、種類の順に保存される。

        Parameters
        ----
        path:str
            保存先ファイルパス。
        """
        if path is None:
            path = os.path.join(
                os.path.dirname(self.path),
                os.path.splitext(f'{os.path.basename(self.path)[0]}.csv'))
        self.gdf.to_csv(path)
        if self.debug:
            print(f'saved to {path}')

    def save_geojson(self, path:str=None) -> None:
        """
        ジオイドモデルをGeoJson形式で保存する。

        Parameters
        ----
        path:str
            GeoJson形式ファイルパス
        """
        if path is None:
            path = os.path.join(
                os.path.dirname(self.path),
                os.path.splitext(f'{os.path.basename(self.path)[0]}.json'))
        self.gdf.to_file(driver='GeoJSON', filename=path)
        if self.debug:
            print(f'saved to {path}')

    def save_geoshp(self, path:str=None) -> None:
        """
        ジオイドモデルをShp形式で保存する。

        Parameters
        ----
        path:str
            GeoJson形式ファイルパス
        """
        if path is None:
            path = os.path.join(
                os.path.dirname(self.path),
                os.path.splitext(f'{os.path.basename(self.path)[0]}.shp'))
        self.gdf.to_file(driver='ESRI Shapefile', filename=path)
        if self.debug:
            print(f'saved to {path}')

class Converter:
    """
    指定されたディレクトリ内の複数のXMLファイル(FG-GML-*.xml)を
    CSV/GeoJSON/Shapeに変換するユーティリティクラス。
    """
    # ターゲットとするファイル名正規表現
    EXPR = 'FG-GML*.xml'

    def __init__(self, data_dir:str='', debug:bool=False) -> None:
        """
        指定されたディレクトリに格納された拡張子(.xml)のファイルパスを
        変換対象とする。

        Parameters
        ----
        data_dir:str=''
            対象とするXMLファイル群が格納されているディレクトリ
        debug:bool=False
            デバッグオプション

        Raises
        ----
        ValueError
            指定されたディレクトリが存在しない場合
        """
        self.debug = debug
        self.data_dir = data_dir
        if not os.path.isdir(self.data_dir):
            raise ValueError(f'{self.data_dir} is not a directory')

        self.paths = [p for p in glob.glob(os.path.join(self.data_dir, self.EXPR))]
        if self.debug:
            print(f'{len(self.paths)} xml files in {self.data_dir}')

    def to_graphs(self, output_dir:str=None) -> None:
        """
        対象のXMLファイルすべての散布図・ヒストグラムファイルを出力する。

        Parameters
        ----
        output_dir:str=None
            出力先ディレクトリ（存在しない場合作成）
        """
        # XMLなし→なにもしない
        if len(self.paths) == 0:
            return

        for path in self.paths:
            if output_dir is None:
                output_dir = os.path.join(
                    os.path.dirname(path), 'out')
            os.makedirs(output_dir, exist_ok=True)
            prefix_path = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(path))[0])
            BasicModel(path, self.debug).get_scatter2d(f'{prefix_path}_2d.png')
            gc.collect()


    def to_all(self, output_dir:str=None) -> None:
        """
        対象のXMLファイルをすべてCSV/Shape/geoJSONファイルに変換する。
        またヒストグラム、散布図も合わせて作成する。

        Parameters
        ----
        output_dir:str=None
            出力先ディレクトリ（存在しない場合作成）
        """
        # XMLなし→なにもしない
        if len(self.paths) == 0:
            return
        for path in self.paths:
            if output_dir is None:
                output_dir = os.path.join(
                    os.path.dirname(path), 'out')
            os.makedirs(output_dir, exist_ok=True)
            prefix_path = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(path))[0])
            model = BasicModel(path, self.debug)
            model.save_csv(f'{prefix_path}.csv')
            model.save_geojson(f'{prefix_path}.json')
            model.save_geoshp(f'{prefix_path}.shp')
            model.get_scatter2d(f'{prefix_path}_2d.png')
            del model
            gc.collect()

if __name__ == '__main__':
    """
    引数にて指定されたデータディレクトリ内のXML(*FG-GML*.xml)をすべて読み込み、
    Geo JSON/Shapeファイルや統計グラフを生成する。
    引数指定された格納先ディレクトリに格納される。
    """
    import argparse
    DATA_ROOT_DIR = os.path.join(os.path.dirname(__file__), 'download')
    parser = argparse.ArgumentParser(description='convert Japan DEM XML files to GeoJSON/Shape files.')
    parser.add_argument('--datadir', type=str, default=DATA_ROOT_DIR, help='DEM data directory path')
    parser.add_argument('--outdir', type=str, default='output', help='output directory path')
    parser.add_argument('--debug', action='store_false', help='print debug lines')
    args = parser.parse_args()

    import time
    elapsed = time.process_time()
    datadirs = glob.glob(args.datadir + '/**/', recursive=True)
    for datadir in datadirs:
        files = glob.glob(datadir + '*FG-GML*.xml')
        if(len(files)> 0):
            conv = Converter(datadir, args.debug)
            conv.to_all(output_dir='output')
        else:
            if args.debug:
                print(f'skip {datadir}')
    elapsed -= time.process_time()
    print(f'elapsed time: {abs(elapsed)} sec')