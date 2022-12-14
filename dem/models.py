# -*- coding: utf-8 -*-
"""
国土交通省国土地理院基盤地図情報数値標高モデルを
扱うためのユーティリティモジュール。
"""
import gc
import os
import csv
import glob
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Point

# ジオイドモデルの地理座標系
DEFAULT_CRS = 'EPSG:6668'

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
    TYPE_NO_DATA = 'no data point'

    def __init__(self, path:str=None, omit_no_data:bool=True, debug:bool=False) -> None:
        """
        基盤地図情報DEM5M/10M XMLファイルを読み込む。

        Parameters
        ----
        path:str=None
            基盤地図情報DEM5m/10m XMLファイルパス
        omit_no_data:bool=True
            真:NO_DATA値を削除する、偽:NO_DATAも含める
        debug:bool=False
            デバッグフラグ（メタ情報表示）
        """
        # パス
        self.path = path

        # デバッグフラグ
        self.debug = debug

        # NO_DATAを削除するか
        self.omit_no_data = omit_no_data

        if self.debug:
            print(f'init path={self.path}, omit_no_data={self.omit_no_data}, debug:{self.debug}')

        # GMLファイルを読み込みメタ情報及びデータを
        # インスタンス変数へ格納
        self.load(self.path)

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

        # 地理座標系
        self.srs_name = root_element.find(
            './/gml:boundedBy', prefix_map).find(
                './/gml:Envelope', prefix_map).get('srsName')

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
        start_point_element = root_element.find('.//gml:startPoint', prefix_map).text.split()

        # データ始点位置、データ終点位置
        if axislavels_element[0] == 'y':
            # 先頭ラベルが 'y' なら座標位置を置換
            self.low =  [int(low_coord[1]),  int(low_coord[0])]
            self.high = [int(high_coord[1]), int(high_coord[0])]
            self.start_point = [int(start_point_element[0]), int(start_point_element[1])]
        else:
            self.low =  [int(low_coord[0]),  int(low_coord[1])]
            self.high = [int(high_coord[0]), int(high_coord[1])]
            self.start_point = [int(start_point_element[1]), int(start_point_element[0])]

        seq_rule_element = root_element.find('.//gml:sequenceRule', prefix_map)
        order_element = seq_rule_element.get('order')

        # メッシュデータの並び方
        self.seq_rule = seq_rule_element.text

        # メッシュデータの方向
        if '-x' in order_element and '-y' in order_element:
            # X(横軸)方向負、Y(縦軸)方向負
            self.order = [-1, -1]
        elif '-x' in order_element and '+y' in order_element:
            # X(横軸)方向負、Y(縦軸)方向正
            self.order = [-1, 1]
        elif '+x' in order_element and '-y' in order_element:
            # X(横軸)方向正、Y(縦軸)方向負
            self.order = [1, -1]
        else:
            # X(横軸)方向正、Y(縦軸)方向正
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
            start_point_element, data_block_element, tupples

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
        print(f'srs name:    {self.srs_name}')
        print(f'description: {self.description}')
        print(f'mesh no:     {self.mesh_no}')
        print(f'mesh type:   {self.mesh_type}')
        print(f'mesh range:  [{self.lower[0]}, {self.lower[1]}] - [{self.upper[0]}, {self.upper[1]}]')
        print(f'mesh position range: [{self.low[0]}, {self.low[1]}] - [{self.high[0]}, {self.high[1]}] sequence: {self.seq_rule}')
        print(f'mesh start position: [{self.start_point[0]}, {self.start_point[1]}]')
        print(f'mesh order:          [{self.order[0]}, {self.order[1]}]')
        print(f'mesh length: x:{len(self.x)}, y:{len(self.y)}, z:{len(self.z)} type:{len(self.t)} uom:{self.uom} (omitted no data: {self.omit_no_data})')

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
        ax.set_xlabel('longitude(degree)',size=10,color='black')
        ax.set_ylabel('latitude(degree)',size=10,color='black')
        ax.set_zlabel('height(m)', size=10, color='black')

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
                # 緯度、経度、標高、種類の順
                writer.writerow([self.y[i], self.x[i], self.z[i], self.t[i]])
        if self.debug:
            print(f'saved csv to {path}')

    def save_geojson(self, path:str, crs:str=DEFAULT_CRS) -> None:
        """
        指定された測地系でGeoJSON形式として保存する。

        Parameters
        ----
        crs:str
            測地系。デフォルトは世界測地系(EPSG:6668)
        """
        self.get_gpd(crs=crs).to_file(driver='GeoJSON', filename=path)
        if self.debug:
            print(f'saved geojson to {path}/{crs}')

    def save_geoshp(self, path:str='geoid.shp', crs:str=DEFAULT_CRS):
        """
        指定された測地系でShp形式で保存する。

        Parameters
        ----
        path:str
            GIS Shape形式ファイルパス
        crs:str
            座標系（デフォルト: 'EPSG:6668'）
        """
        self.get_gpd(crs=crs).to_file(driver='ESRI Shapefile', filename=path)
        if self.debug:
            print(f'saved shp to {path}/{crs}')

    def get_gpd(self, crs:str=DEFAULT_CRS) -> gpd.GeoDataFrame:
        """
        DEMデータをGeoDataFrame オブジェクトとして取得する。
        geometry:座標(経度、緯度）、height:標高、type:種類

        Parameters
        ----
        crs:str
            座標系（デフォルト EPSG:6668）
        
        Returns
        ----
        gpd.GeoDataFrame
            ジオイドモデル(標高:'height'属性、種類：'type'属性)
        """

        geometry = []
        total = len(self.x)
        for i in range(total):
            # Point(経度, 緯度)
            geometry.append(Point(self.x[i], self.y[i]))
        return gpd.GeoDataFrame({'height':self.z, 'type':self.t, 'geometry':geometry}, crs=crs)

    def get_xyzt(self, z:list, t:list) -> tuple[list, list, list, list]:
        """
        インスタンス変数に格納されているメタ情報をもとに
        計測データなしを除外した経度(X)リスト、緯度(Y)リスト、標高リスト、DEM種別リストを生成する。

        Parameters
        ----
        z:list
            標高リスト
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
        # 先頭データなし領域にNO_DATAを追加
        if self.debug:
            print(f'len z:{len(z)}, t:{len(t)}')
        # 開始位置までNO_DATAで埋める
        no_data_size = self.start_point[0] * (abs(self.high[0] - self.low[0]) + 1) + self.start_point[1]
        for _ in range(no_data_size):
            z.insert(0, self.NO_DATA)
            t.insert(0, self.TYPE_NO_DATA)
        if no_data_size > 0 and self.debug:
            print(f'inserted {no_data_size} no_data')
        if self.debug:
            print(f'len z:{len(z)}, t:{len(t)}')

        # X方向（横軸）
        start_x_position, end_x_position = self.lower[0], self.upper[0]
        start_x, end_x, delta_x, total_x = self._get_range(self.order[0], self.low[0], self.high[0])
        if self.debug:
            print(f'x start_pos:{start_x_position}, end_pos:{end_x_position}')
            print(f'x start:{start_x}, end:{end_x}, delta:{delta_x}, total:{total_x}')
        # Y方向（縦軸）
        start_y_position, end_y_position = self.lower[1], self.upper[1]
        start_y, end_y, delta_y, total_y = self._get_range(self.order[1], self.low[1], self.high[1])
        if self.debug:
            print(f'y start_pos:{start_y_position}, end_pos:{end_y_position}')
            print(f'y start:{start_y}, end:{end_y}, delta:{delta_y}, total:{total_y}')

        # 標高点数カウンタ
        index_z = 0
        # 経度、緯度、標高、種類
        _x, _y, _z, _t = [], [], [], []
        # Y軸ループ
        for index_y in range(start_y, end_y, delta_y):
            # Y軸座標値
            pos_y = self._get_position(index_y, total_y, start_y_position, end_y_position)
            # X軸ループ
            for index_x in range(start_x, end_x, delta_x):
                # X軸座標値
                pos_x = self._get_position(index_x, total_x, start_x_position, end_x_position)
                # Z軸座標値
                pos_z = z[index_z]
                
                # NO_DATAの場合リストに入れない or NO_DATAではない
                if not self.omit_no_data or pos_z > self.NO_DATA:
                    _x.append(pos_x) # 経度
                    _y.append(pos_y) # 緯度
                    _z.append(pos_z) # 標高
                    _t.append(t[index_z]) # 種類（日本語）
                elif self.debug:
                    print(f'omitted (x, y, z, t) = ({pos_x}, {pos_y}, {pos_z}, {t[index_z]})')

                index_z += 1
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

    def __init__(self, data_dir:str='', omit_no_data:bool=True, debug:bool=False) -> None:
        """
        指定されたディレクトリに格納された拡張子(.xml)のファイルパスを
        変換対象とする。

        Parameters
        ----
        data_dir:str=''
            対象とするXMLファイル群が格納されているディレクトリ
        omit_no_data:bool=True
            NO_DATAを無視するかどうか
        debug:bool=False
            デバッグオプション

        Raises
        ----
        ValueError
            指定されたディレクトリが存在しない場合
        """
        self.debug = debug
        self.data_dir = data_dir
        self.omit_no_data = omit_no_data
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
            csv_path = f'{os.path.join(output_dir, os.path.splitext(os.path.basename(path))[0])}.csv'
            mesh = MeshModel(path=path, omit_no_data=self.omit_no_data, debug=self.debug)
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
            json_path = f'{os.path.join(output_dir, os.path.splitext(os.path.basename(path))[0])}.json'
            mesh = MeshModel(path=path, omit_no_data=self.omit_no_data, debug=self.debug)
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
            shp_path = f'{os.path.join(output_dir, os.path.splitext(os.path.basename(path))[0])}.shp'
            mesh = MeshModel(path=path, omit_no_data=self.omit_no_data, debug=self.debug)
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
            mesh = MeshModel(path=path, omit_no_data=self.omit_no_data, debug=self.debug)
            mesh.get_histgram(f'{os.path.join(output_dir, prefix)}_hist.png')
            mesh.get_scatter3d(f'{os.path.join(output_dir, prefix)}_3d.png')
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
            mesh = MeshModel(path=path, omit_no_data=self.omit_no_data, debug=self.debug)
            mesh.get_histgram(f'{os.path.join(output_dir, prefix)}_hist.png')
            mesh.get_scatter3d(f'{os.path.join(output_dir, prefix)}_3d.png')
            mesh.save_csv(f'{os.path.join(output_dir, prefix)}.csv')
            mesh.save_geojson(f'{os.path.join(output_dir, prefix)}.json')
            mesh.save_geoshp(f'{os.path.join(output_dir, prefix)}.shp')
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
    parser.add_argument('--omit_no_data', action='store_true', help='omit no data records')
    parser.add_argument('--debug', action='store_true', help='print debug lines')
    args = parser.parse_args()

    import time
    elapsed = time.process_time()
    datadirs = glob.glob(args.datadir + '/**/', recursive=True)
    for datadir in datadirs:
        files = glob.glob(datadir + '*.xml')
        if(len(files)> 0):
            #if args.debug:
            #    print(f'Converter args: datadir={datadir}, outdir={args.outdir}, omit_no_data={args.omit_no_data}, debug={args.debug}')
            conv = Converter(data_dir=datadir, omit_no_data=args.omit_no_data, debug=args.debug)
            conv.to_all(output_dir=args.outdir)
        else:
            if args.debug:
                print(f'skip {datadir}')
    elapsed -= time.process_time()
    print(f'elapsed time: {abs(elapsed)} sec')