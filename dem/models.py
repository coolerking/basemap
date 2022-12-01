# -*- coding: utf-8 -*-
"""
国土交通省国土地理院基盤地図情報数値標高モデルを
扱うためのユーティリティモジュール。
"""
import gc
import os
import csv
import glob
#import pathlib
#import collections
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Point


class MeshModel:
    """
    国土交通省国土地理院基盤地図情報数値標高モデルダウンロードファイル(GML形式)
    をよみこみ、GISデータとして変換するためのクラス。
    1インスタンスで1ダウンロードファイル(1メッシュ)を扱う。
    """
    
    """
    データなし時の数値
    """
    NO_DATA = -9999.0

    def __init__(self, path:str=None, debug:bool=False) -> None:
        """
        
        """
        # デバッグフラグ
        self.debug = debug

        # GMLファイルを読み込みメタ情報及びデータを
        # インスタンス変数へ格納
        self.load(path)

        # メタ情報表示
        if self.debug:
            self.show_meta()
    
    def load(self, path:str) -> None:
        """
        国土交通省国土地理院基盤地図情報数値標高モデルダウンロードファイル(GML形式)
        を読み込み、インスタンス変数へ格納する。
        全件データ要素をインスタンス変数へ格納するため、メモリ不足に注意。

        Parameters
        ----
        path:str        読み込み対象ファイルパス
        """
        # XMLファイルパス
        self.path = path

        # XMLスキーママッピング
        prefix_map = {
           'gml': 'http://www.opengis.net/gml/3.2',
            '': 'http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema',
            'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
        }

        # XMLファイルのパース
        root_element = ET.parse(path).getroot()

        # データ名称
        self.name = root_element.find('.//gml:name', prefix_map).text
        # データの説明
        self.description = root_element.find('.//gml:description', prefix_map).text
        # メッシュ番号
        self.mesh_no = root_element.find('.//mesh', prefix_map).text
        # データ種類
        self.mesh_type = root_element.find('.//type', prefix_map).text

        # メッシュ矩形左下頂点位置
        lower_element = root_element.find('.//gml:lowerCorner', prefix_map).text.split()
        #self.lower = [float(lower_element[0]), float(lower_element[1])]
        self.lower = [float(lower_element[1]), float(lower_element[0])]
        # メッシュ矩形右上頂点位置
        upper_element = root_element.find('.//gml:upperCorner', prefix_map).text.split()
        #self.upper = [float(upper_element[0]), float(upper_element[1])]
        self.upper = [float(upper_element[1]), float(upper_element[0])]

        grid_element = root_element.find('.//gml:Grid', prefix_map)
        low_coord = grid_element.find('.//gml:low', prefix_map).text.split()
        high_coord = grid_element.find('.//gml:high', prefix_map).text.split()
        axislavels_element = grid_element.find('.//gml:axisLabels', prefix_map).text.split()

        # データ始点位置、データ終点位置
        if axislavels_element[0] == 'y':
            # 先頭ラベルが 'y' なら座標位置を置換
            self.low =  [self.low[1],  self.low[0]]
            self.high = [self.high[1], self.high[0]]
        else:
            self.low =  [int(low_coord[0]),  int(low_coord[1])]
            self.high = [int(high_coord[0]), int(high_coord[1])]

        seq_rule_element = root_element.find('.//gml:sequenceRule', prefix_map)
        order_element = seq_rule_element.get('order')

        # メッシュデータの並び方
        self.seq_rule = seq_rule_element.text

        # メッシュデータの方向
        if '-x' in order_element and '-y' in order_element:
            # X(緯度、横軸)方向負、Y(経度、縦軸)方向負
            self.order = [-1, -1]
        elif '-x' in order_element and '+y' in order_element:
            # X(緯度、横軸)方向負、Y(経度、縦軸)方向正
            self.order = [-1, 1]
        elif '+x' in order_element and '-y' in order_element:
            # X(緯度、横軸)方向正、Y(経度、縦軸)方向負
            self.order = [1, -1]
        else:
            # X(緯度、横軸)方向正、Y(経度、縦軸)方向正
            self.order = [1, 1]

        data_block_element = root_element.find('.//gml:DataBlock', prefix_map)

        # データ要素の種類
        self.uom = data_block_element.find('.//gml:QuantityList', prefix_map).get('uom')

        # データ全要素
        tupples = data_block_element.find('.//gml:tupleList', prefix_map).text.split()

        # 各要素の標高、各要素の種別
        z, t = [], []
        for tupple in tupples:
            point = tupple.strip().split(',')
            t.append(point[0])
            z.append(float(point[1]))

        del root_element, lower_element, upper_element, \
            grid_element, axislavels_element, seq_rule_element, order_element, \
            data_block_element, tupples

        print(f'path:        {self.path}')
        print(f'name:        {self.name}')
        print(f'description: {self.description}')
        print(f'mesh no:     {self.mesh_no}')
        print(f'mesh type:   {self.mesh_type}')
        print(f'mesh range:  [{self.lower[0]}, {self.lower[1]}] - [{self.upper[0]}, {self.upper[1]}]')
        print(f'mesh position range: [{self.low[0]}, {self.low[1]}] - [{self.high[0]}, {self.high[1]}] sequence: {self.seq_rule}')
        print(f'mesh order:          [{self.order[0]}, {self.order[1]}]')
        print(f'mesh length: z:{len(z)} type:{len(t)} uom:{self.uom}')

        # メタ情報から緯度経度リストを生成し
        # インスタンス変数へ格納
        (self.x, self.y, self.z, self.t) = self.get_xyzt(z, t)

        # ガーベージコレクション
        del z, t
        gc.collect()

    def show_meta(self):
        """
        メタ情報を表示する。
        """
        print(f'path:        {self.path}')
        print(f'name:        {self.name}')
        print(f'description: {self.description}')
        print(f'mesh no:     {self.mesh_no}')
        print(f'mesh type:   {self.mesh_type}')
        print(f'mesh range:  [{self.lower[0]}, {self.lower[1]}] - [{self.upper[0]}, {self.upper[1]}]')
        print(f'mesh position range: [{self.low[0]}, {self.low[1]}] - [{self.high[0]}, {self.high[1]}] sequence: {self.seq_rule}')
        print(f'mesh order:          [{self.order[0]}, {self.order[1]}]')
        print(f'mesh length: x:{len(self.x)}, y:{len(self.y)}, z:{len(self.z)} type:{len(self.t)} uom:{self.uom}')

    def get_histgram(self, path:str=None):
        """
        標高値のヒストグラムを表示・保存する。
        負値(値なし相当値-9999.0含む)は対象外とする。

        Parameters
        ----
        path:str
            保存先ファイルパス、指定しない場合表示される。
        """

        # フォントファミリ指定
        #plt.rcParams['font.family'] = 'Meiryo'

        # figure の作成
        fig = plt.figure(figsize=(8, 6))

        # subplot の追加
        ax = fig.add_subplot()

        # タイトルの作成
        ax.set_title(self.path, size=10)
        ax.hist(self.z, range(0, int(np.max(np.array(self.z)))+1))
        
        # 保存先パスが定義されていない場合
        if path is None:
            # グラフを表示
            plt.show()
        else:
            plt.savefig(path)
            if self.debug:
                print(f'saved histgram to {path}')

    def get_scatter3d(self, path:str=None) -> None:
        """
        3次元散布図を表示・保存する

        Parameters
        ----
        path:str
            3次元散布図保存先ファイルパス、指定なしの場合表示させる
        """
        # フォントファミリ指定
        #plt.rcParams['font.family'] = 'Meiryo'

        # figure の作成
        fig = plt.figure(figsize=(8, 6))

        # subplot の追加
        ax = fig.add_subplot(projection='3d')

        # タイトルの作成
        ax.set_title(self.path, size=10)
 
        # 軸ラベルのサイズと色を設定
        ax.set_xlabel('latitude(degree)',size=10,color='black')
        ax.set_ylabel('longitude(degree)',size=10,color='black')
        ax.set_zlabel('height(m)', size=10, color='black')

        # リストx:緯度、リストy:経度、リストz:ジオイド高 に変換
        #(x, y, z) = self.convert_xyz()

        # 散布図を描画
        ax.scatter(self.x, self.y, self.z, s=1, c='red')

        # 保存先指定なし
        if path is None:
            # 散布図を表示
            plt.show()
        else:
            # 散布図を保存
            plt.savefig(path)
            if self.debug:
                print(f'saved 3d scatter to {path}')

    def save_csv(self, path:str) -> None:
        """
        CSV形式ファイルとして保存する。
        緯度、経度、標高、種類の順に保存される。

        Parameters
        ----
        path:str
            保存先ファイルパス。
        """
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            for i in range(len(self.z)):
                writer.writerow([self.x[i], self.y[i], self.z[i], self.t[i]])
        if self.debug:
            print(f'saved csv to {path}')

    def save_geojson(self, path:str, crs:str='EPSG:4326') -> None:
        """
        指定された測地系でGeoJSON形式として保存する。

        Parameters
        ----
        crs:str
            測地系。デフォルトは世界測地系(EPSG:4326)
        """
        self.get_gpd(crs=crs).to_file(driver='GeoJSON', filename=path)
        if self.debug:
            print(f'saved geojson to {path}')

    def save_geoshp(self, path:str='geoid.shp', crs:str='EPSG:4326'):
        """
        指定された測地系でShp形式で保存する。

        Parameters
        ----
        path:str
            GIS Shape形式ファイルパス
        crs:str
            座標系（デフォルト: 'EPSG:4326'）
        """
        self.get_gpd(crs=crs).to_file(driver='ESRI Shapefile', filename=path)
        if self.debug:
            print(f'saved shp to {path}')

    def get_gpd(self, crs:str='EPSG:4326') -> gpd.GeoDataFrame:
        """
        DEMデータをGeoDataFrame オブジェクトとして取得する。
        データなし(-9999.0)である座標は削除済みオブジェクトとなる。

        Parameters
        ----
        crs:str
            座標系（デフォルト EPSG:4326）
        
        Returns
        ----
        gpd.GeoDataFrame
            ジオイドモデル(標高:'height'属性、種類：'type'属性)
        """

        geometry = []
        total = len(self.x)
        for i in range(total):
            # Point(経度, 緯度)
            geometry.append(Point(self.y[i], self.x[i]))
        return gpd.GeoDataFrame({'height':self.z, 'type':self.t, 'geometry':geometry}, crs=crs)

    def get_xyzt(self, z:list, t:list) -> tuple[list, list, list, list]:
        """
        インスタンス変数に格納されているメタ情報をもとに
        計測データなしを除外した経度(X)リスト、緯度(Y)リスト、標高リスト、DEM種別リストを生成する。
        Parameters
        ----
        z:list
            標高リスト(NO_DATA含む)
        t:list
            DEM種別リスト

        Returns
        ----
        tuple(list, list, list, list)
            経度リスト（単位：度）
            緯度リスト（単位：度）
            標高リスト（単位：ｍ）
            DEM種別リスト（文字列）
        """
        start_x_position, end_x_position = self.lower[0], self.upper[0]
        start_x, end_x, delta_x, total_x = self._get_range(self.order[0], self.low[0], self.high[0])
        if self.debug:
            print(f'x start_pos:{start_x_position}, end_pos:{end_x_position}')
            print(f'x start:{start_x}, end:{end_x}, delta:{delta_x}, total:{total_x}')

        start_y_position, end_y_position = self.lower[1], self.upper[1]
        start_y, end_y, delta_y, total_y = self._get_range(self.order[1], self.low[1], self.high[1])
        if self.debug:
            print(f'y start_pos:{start_y_position}, end_pos:{end_y_position}')
            print(f'y start:{start_y}, end:{end_y}, delta:{delta_y}, total:{total_y}')
        print(f'z:{len(z)}, t:{len(t)}')
        index_z = 0
        _x, _y, _z, _t = [], [], [], []
        for index_y in range(start_y, end_y, delta_y):
            pos_y = self._get_position(index_y, total_y, start_y_position, end_y_position)
            for index_x in range(start_x, end_x, delta_x):
                pos_x = self._get_position(index_x, total_x, start_x_position, end_x_position)
                pos_z = z[index_z]
                _x.append(pos_x)
                _y.append(pos_y)
                _z.append(pos_z)
                _t.append(t[index_z])
                index_z = index_z + 1
                # XMLファイルによってはメタ情報記載のデータ数より前に終了するものがある
                # ex) FG-GML-5339-15-04-DEM5A-20161001.xml
                # このため早めにデータが無くなる場合はbreakして止めている
                if index_z >= len(z):
                    if self.debug:
                        print(f'break xyzt loop: index_z={index_z} arrived to out of range/ len(z)={len(z)}')
                    # break inner loop
                    break
            # loop out without break
            else:
                continue
            # break outer loop
            break
        if self.debug:
            print(f'len longitude:{len(_x)} latitude:{len(_y)} height:{len(_z)} type:{len(_t)}')
        return (_x, _y, _z, _t)

    def _get_range(self, order:int, low:int, high:int) -> tuple[int, int, int, int]:
        """
        対象位置ループのためのrange引数情報を返却する。
    
        Parameters
        ----
        order:int
            方向(+1:正方向、-1:負方向)
        low:int
            開始位置
        high:int
            終了位置
    
        Returns
        ----
        tuple(int, int, int, int)
            開始インデックス値(range第1引数)
            終了インデックス値(range第2引数)
            加算値(range第3引数)
            合計件数
        """
        total = abs(high - low) + 1
        if order == 1:
            return (low, (high + 1), 1, total)
        else:
            return (high, (low - 1), -1, total)

    def _get_position(self, index:int, total:int, start_position:float, end_position:float) -> float:
        """
        引数で指定された、対象位置の座標値（単位：度）を返却する。

        Parameters
        ----
        index:int
            対象位置
        total:int
            全件数
        start_position:float
            開始座標値（単位：度）
        end_latitude:float
            終了座標値（単位：度）
    
        Returns
        ----
        float
            対象位置の座標値（単位：度）
        """
        if index == 0:
            return start_position
        elif index == total -1:
            return end_position
        else:
            return float(index) * (end_position - start_position) / float(total) + start_position

class Converter:
    """
    指定されたディレクトリ内の複数のXMLファイルを
    CSV/GeoJSON/Shapeに変換するユーティリティクラス。
    """

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

        self.paths = [p for p in glob.glob(os.path.join(self.data_dir, '*.xml'))]
        if self.debug:
            print(f'{len(self.paths)} xml files in {self.data_dir}')

    def to_csv(self, output_dir:str='out') -> None:
        """
        対象のXMLファイルをすべてCSVファイルに変換する。

        Parameters
        ----
        output_dir:str='out'
            出力先ディレクトリ（存在しない場合作成）
        """
        os.makedirs(output_dir, exist_ok=True)
        for path in self.paths:
            csv_path = os.path.join(
                output_dir, os.path.splitext(os.path.basename(path))[0] + '.csv')
            mesh = MeshModel(path, self.debug)
            mesh.save_csv(csv_path)
            del mesh
            gc.collect()

    def to_geojson(self, output_dir:str='out') -> None:
        """
        対象のXMLファイルをすべてGeo JSONファイルに変換する。

        Parameters
        ----
        output_dir:str='out'
            出力先ディレクトリ（存在しない場合作成）
        """
        os.makedirs(output_dir, exist_ok=True)
        for path in self.paths:
            json_path = os.path.join(
                output_dir, os.path.splitext(os.path.basename(path))[0] + '.json')
            mesh = MeshModel(path, self.debug)
            mesh.save_geojson(json_path)
            del mesh
            gc.collect()

    def to_shp(self, output_dir:str='out') -> None:
        """
        対象のXMLファイルをすべてShapeファイルに変換する。

        Parameters
        ----
        output_dir:str='out'
            出力先ディレクトリ（存在しない場合作成）
        """
        os.makedirs(output_dir, exist_ok=True)
        for path in self.paths:
            shp_path = os.path.join(
                output_dir, os.path.splitext(os.path.basename(path))[0] + '.shp')
            mesh = MeshModel(path, self.debug)
            mesh.save_geoshp(shp_path)
            del mesh
            gc.collect()

    def to_graphs(self, output_dir:str='out') -> None:
        """
        対象のXMLファイルすべての散布図・ヒストグラムファイルを出力する。

        Parameters
        ----
        output_dir:str='out'
            出力先ディレクトリ（存在しない場合作成）
        """
        os.makedirs(output_dir, exist_ok=True)
        for path in self.paths:
            prefix = os.path.splitext(os.path.basename(path))[0]
            mesh = MeshModel(path, self.debug)
            mesh.get_histgram(os.path.join(output_dir, prefix + '_hist.png'))
            mesh.get_scatter3d(os.path.join(output_dir, prefix + '_3d.png'))
            del mesh
            gc.collect()

    def to_all(self, output_dir:str='out') -> None:
        """
        対象のXMLファイルをすべてCSV/Shape/geoJSONファイルに変換する。
        またヒストグラム、散布図も合わせて作成する。

        Parameters
        ----
        output_dir:str='out'
            出力先ディレクトリ（存在しない場合作成）
        """
        os.makedirs(output_dir, exist_ok=True)
        for path in self.paths:
            if self.debug:
                print(f'** {path} **')
            prefix = os.path.splitext(os.path.basename(path))[0]
            mesh = MeshModel(path, self.debug)
            mesh.get_histgram(os.path.join(output_dir, prefix + '_hist.png'))
            mesh.get_scatter3d(os.path.join(output_dir, prefix + '_3d.png'))
            mesh.save_csv(os.path.join(output_dir, prefix + '.csv'))
            mesh.save_geojson(os.path.join(output_dir, prefix + '.json'))
            mesh.save_geoshp(os.path.join(output_dir, prefix + '.shp'))
            del mesh
            gc.collect()

if __name__ == '__main__':
    """
    引数にて指定されたデータディレクトリ内のXMLをすべて読み込み、
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

    datadirs = glob.glob(args.datadir + '/**/', recursive=True)
    for datadir in datadirs:
        files = glob.glob(datadir + '*.xml')
        if(len(files)> 0):
            conv = Converter(datadir, args.debug)
            conv.to_all(output_dir='output')
        else:
            if args.debug:
                print(f'skip {datadir}')