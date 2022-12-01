# -*- coding: utf-8 -*-
"""
ジオイドモデル「日本のジオイド」データを操作するための管理クラスモジュール。

注意：
使用する前に、ジオイドモデル「日本のジオイド」をダウンロードしておく必要があります。
https://fgd.gsi.go.jp/download/geoid.php より
ASCII形式のファイルをダウンロード・展開し、
拡張子.asc で保存されているジオイドモデルのパスをコンストラクタ引数 path に
指定してください。
"""
import os
import re
import csv
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from typing import Tuple

import matplotlib.pyplot as plt

# ダウンロードしたジオイドモデルパス
# 2022/11/30 ダウンロード
GEOID_FILENAME = 'gsigeo2011_ver2_1.asc'
GEOID_PATH = os.path.join('download', GEOID_FILENAME)

class GeoidModel:
    """
    ジオイドモデル「日本のジオイド」データを操作するための管理クラス。
    """

    """
    ASCII版日本のジオイドデータの値なし時に格納されている数値
    """
    NO_DATA = 999.0

    """
    緯度差分(1分)
    """
    DELTA_LATITUDE = 1.0 / 60.0

    """
    経度差分(1.5分)
    """
    DELTA_LONGITUDE = 1.5 / 60.0

    """
    上限緯度（単位：度）
    """
    MAX_LATITUDE = 50.0

    """
    上限経度（単位：度）
    """
    MAX_LONGITUDE = 150.0

    def __init__(self, path:str=GEOID_PATH, debug:bool=False) -> None:
        """
        日本のジオイド データファイルを読み込み、
        ジオイド高計算のために必要なデータをクラス変数に格納する。

        Parameters
        ----
        path:str=GEOID_PATH
            日本のジオイド データファイルパス
        debug:bool=False
            デバッグオプション
        """
        # ジオイドデータファイルパス
        self.path = path
        # デバッグオプション
        self.debug = debug

        # ジオイドデータ(ASCII形式)の読み込み
        with open(self.path, 'r', encoding='utf-8') as f:  # ファイルを開く
            # 先頭行データの要素分割
            line = f.readline().strip()
            tokens = re.split(' +', line)

            # メタ情報をクラス変数へ格納
            self.glamn = float(tokens[0]) # 南端の緯度(北緯20度)
            self.glomn = float(tokens[1]) # 西端の経度(東経120度)
            self.dgla =  float(tokens[2]) # 緯度間隔(度)
            self.dglo =  float(tokens[3]) # 経度間隔(度)
            self.nla =   int(float(tokens[4])) # 緯線の個数/Y:1801
            self.nlo =   int(float(tokens[5])) # 経線の個数/X:1201
            self.ikind = int(float(tokens[6])) # フォーマット識別子
            self.vern =  str(tokens[7]) # データのバージョン
            self._revise_delta() # メタ情報だと精度が低いので算出しなおす

            idx = 0 # rowに格納した要素数 0,1,2,.. ,(nlo-1)
            row = [] # 1行のデータを要素分割したもの
            self.rows = [] # 各行の row を読み込んだ順に格納 
        
            # 1行づつ読み込み
            for line in f.readlines():
                # 読み込んだ行を要素分割(要素はジオイド高:float値)
                tokens = re.split(' +', line.strip())

                # 要素を1件づつ処理
                for token in tokens:
                    # リストrowへ要素を1件追加
                    row.append(float(token))
                    # インデックスを加算
                    idx = idx + 1

                    # インデックスが経度要素数上限を超えていない場合
                    if idx < self.nlo:
                        # 継続
                        continue
                    # インデックスが経度要素数上限を超える場合
                    else:
                        # インデックスを初期化
                        idx = 0
                        # リストrowをリストrowsへ格納
                        self.rows.append(row)
                        # リストrowは次の経度のジオイド高を格納するために初期化
                        row = []

        if self.debug:
            self.show_meta()

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
        print(f'path:  {self.path}')
        print(f'glamn: {self.glamn}')
        print(f'glomn: {self.glomn}')
        print(f'dgla:  {self.dgla}')
        print(f'dglo:  {self.dglo}')
        print(f'nla:   {self.nla}, rows latitude  length:({len(self.rows)})')
        print(f'nlo:   {self.nlo}, rows longitude length:({len(self.rows[0])})')
        print(f'ikind: {self.ikind}')
        print(f'vern:  {self.vern}')

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

    def get_scatter3d(self, path:str=None) -> None:
        """
        日本のジオイドデータを3次元散布図に変換する。

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
        ax.set_title('Geoid Japan2011 Ver2.1', size=10)
 
        # 軸ラベルのサイズと色を設定
        ax.set_xlabel('latitude(degrees)',size=10,color='black')
        ax.set_ylabel('longitude(degrees)',size=10,color='black')
        ax.set_zlabel('geoid height(m)', size=10, color='black')

        # リストx:緯度、リストy:経度、リストz:ジオイド高 に変換
        (x, y, z) = self.convert_xyz()

        # 散布図を描画
        ax.scatter(x, y, z, s=1, c='red')

        # 保存先指定なし
        if path is None:
            # 散布図を表示
            plt.show()
        else:
            # 散布図を保存
            plt.savefig(path)
            if self.debug:
                print(f'saved 3d scatter to {path}')

    def get_scatter2d(self, path:str=None) -> None:
        """
        日本のジオイドデータを2次元散布図に変換する。

        Parameters
        ----
        path:str
            2次元散布図保存先ファイルパス、指定なしの場合は表示
        """
        # フォントファミリ指定
        #plt.rcParams['font.family'] = 'Meiryo'

        # figure の作成
        fig = plt.figure(figsize=(8, 6))

        # subplot の追加
        ax = fig.add_subplot()

        # タイトルの作成
        ax.set_title('Geoid Japan2011 Ver2.1', size=10)
 
        # 軸ラベルのサイズと色を設定
        ax.set_xlabel('latitude(degrees)',size=10,color='black')
        ax.set_ylabel('longitude(degrees)',size=10,color='black')

        # リストx:緯度、リストy:経度、リストz:ジオイド高 に変換
        (x, y, _) = self.convert_xyz()

        # 散布図を描画
        ax.scatter(x, y, s=1, c='green')

        # 保存先指定なし
        if path is None:
            # 散布図を表示
            plt.show()
        else:
            # 散布図を保存
            plt.savefig(path)
            if self.debug:
                print(f'saved 2d scatter to {path}')

    def _revise_delta(self):
        """
        緯度・経度差分メタ情報をマニュアル情報から算出し
        クラス変数(dgla, dglo)を更新する。
        """
        self.dgla = self.DELTA_LATITUDE
        self.dglo = self.DELTA_LONGITUDE


    def to_csv(self, path:str=None) -> None:
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
                os.path.splitext(os.path.basename(self.path))[0] + '.csv')

        # リストx:緯度、リストy:経度、リストz:ジオイド高 に変換
        (x, y, z) = self.convert_xyz()

        total = len(x)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            for i in range(total):
                writer.writerow([x[i], y[i], z[i]])

        if self.debug:
            print(f'saved to {path}')

    def to_geojson(self, path:str=None, crs:str='EPSG:4326'):
        """
        ジオイドモデルをGeoJson形式で保存する。

        Parameters
        ----
        path:str
            GeoJson形式ファイルパス
        crs:str
            座標系（デフォルト: 'EPSG:4326'）
        """
        if path is None:
            path = os.path.join(
                os.path.dirname(self.path),
                os.path.splitext(os.path.basename(self.path))[0] + '.json')
        self.get_gpd(crs=crs).to_file(driver='GeoJSON', filename=path)

    def to_geoshp(self, path:str=None, crs:str='EPSG:4326'):
        """
        ジオイドモデルをShp形式で保存する。

        Parameters
        ----
        path:str
            GeoJson形式ファイルパス
        crs:str
            座標系（デフォルト: 'EPSG:4326'）
        """
        if path is None:
            path = os.path.join(
                os.path.dirname(self.path),
                os.path.splitext(os.path.basename(self.path))[0] + '.shp')
        self.get_gpd(crs=crs).to_file(driver='ESRI Shapefile', filename=path)

    def to_graphs(self, output_dir:str=None) -> None:
        """
        対象のXMLファイルすべての散布図・ヒストグラムファイルを出力する。

        Parameters
        ----
        output_dir:str=None
            出力先ディレクトリ（存在しない場合作成）
        """
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(self.path), 'out')
        os.makedirs(output_dir, exist_ok=True)
        prefix_path = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(self.path))[0])
        self.get_histgram(prefix_path + '_hist.png')
        self.get_scatter2d(prefix_path + '_2d.png')
        self.get_scatter3d(prefix_path + '_3d.png')


    def to_all(self, output_dir:str=None) -> None:
        """
        対象のXMLファイルをすべてCSV/Shape/geoJSONファイルに変換する。
        またヒストグラム、散布図も合わせて作成する。

        Parameters
        ----
        output_dir:str=None
            出力先ディレクトリ（存在しない場合作成）
        """
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(self.path), 'out')
        os.makedirs(output_dir, exist_ok=True)
        prefix_path = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(self.path))[0])
        self.to_csv(prefix_path + '.csv')
        self.to_geojson(prefix_path + '.json')
        self.to_geoshp(prefix_path + '.shp')
        self.get_histgram(prefix_path + '_hist.png')
        self.get_scatter2d(prefix_path + '_2d.png')
        self.get_scatter3d(prefix_path + '_3d.png')


    def interpolate(self, latitude:float, longitude:float) -> float:
        """
        内挿計算により指定された緯度・経度（単位：度）のジオイド高を算出する。
    
        Parameters
        ----
        latitude:float
            計算対象の緯度（北緯、単位：度）
        longitude:float
            計算対象の経度（整形、単位：度）

        Returns
        ----
        float
            ジオイド高（単位：メートル）

        Raises
        ----
        ValueError
            ジオイドデータ範囲外を指定された場合
        """
        # 緯度インデックス算出
        (low_lat_idx, up_lat_idx) = self._get_latitude_index(latitude)
        # 経度インデックス算出
        (low_lon_idx, up_lon_idx) = self._get_longitude_index(longitude)
        if low_lat_idx == up_lat_idx:
            if low_lon_idx == up_lon_idx:
                return self.rows[low_lat_idx][low_lon_idx]
            else:
                u = (longitude - self._get_longitude(low_lon_idx)) / \
                    (self._get_longitude(up_lon_idx) - self._get_longitude(low_lon_idx))
                return u * (self.rows[low_lat_idx][up_lon_idx] - self.rows[low_lat_idx][low_lon_idx]) + \
                    self.rows[low_lat_idx][low_lon_idx]
        else:
            t = (latitude  - self._get_latitude( low_lat_idx)) / \
                (self._get_latitude( up_lat_idx) - self._get_latitude( low_lat_idx))
            if low_lon_idx == up_lon_idx:
                return t * (self.rows[up_lat_idx][low_lon_idx] - self.rows[low_lat_idx][low_lon_idx]) + \
                    self.rows[low_lat_idx][low_lon_idx]
            else:
                u = (longitude - self._get_longitude(low_lon_idx)) / \
                    (self._get_longitude(up_lon_idx) - self._get_longitude(low_lon_idx))
                return (1 - t) * (1 - u) * self.rows[low_lat_idx][low_lon_idx] + \
                    (1 - t) * u       * self.rows[low_lat_idx][ up_lon_idx] + \
                    t       * (1 - u) * self.rows[ up_lat_idx][low_lon_idx] + \
                    t       * u       * self.rows[ up_lat_idx][ up_lon_idx]

    def interpolate_dms(self, lat_d:int, lat_m:int, lat_s:float, lon_d:int, lon_m:int, lon_s:float) -> float:
        """
        内挿計算により指定された緯度・経度（単位:度分秒）のジオイド高を算出する。
    
        Parameters
        ----
        lat_d:int
            計算対象の緯度（北緯、単位：度）
        lat_m:int
            計算対象の緯度（北緯、単位：分）
        lat_s:float
            計算対象の緯度（北緯、単位：秒）
        lon_d:int
            計算対象の経度（西経、単位：度）
        lon_m:int
            計算対象の経度（西経、単位：分）
        lon_s:float
            計算対象の経度（西経、単位：秒）

        Returns
        ----
        float
            ジオイド高（単位：メートル）

        Raises
        ----
        ValueError
            ジオイドデータ範囲外を指定された場合
        """
        latitude = GeoidModel.to_degree(lat_d, lat_m, lat_s)
        longitude = GeoidModel.to_degree(lon_d, lon_m, lon_s)
        return self.interpolate(latitude, longitude)

    def _get_latitude(self, index:int) -> float:
        """
        緯度インデックス値から緯度（単位：度）に変換する。

        Parameters
        ----
        index:int
            緯度インデックス値
        
        Returns
        ----
        float
            緯度（単位：度）
        
        Raises
        ----
        ValueError
            緯度インデックスが範囲外の場合
        """
        if index < 0 or self.nla <= index:
            raise ValueError(f'latitude index:({index}) is out of range')
        return self.glamn + \
            float(index) * (self.MAX_LATITUDE - self.glamn) / float(self.nla - 1)

    def _get_longitude(self, index:int) -> float:
        """
        経度インデックス値から経度（単位：度）に変換する。

        Parameters
        ----
        index:int
            経度インデックス値
        
        Returns
        ----
        float
            経度（単位：度）
        
        Raises
        ----
        ValueError
            経度インデックスが範囲外の場合
        """
        if index < 0 or self.nlo <= index:
            raise ValueError(f'longitude index:({index}) is out of range')
        return self.glomn + \
            float(index) * (self.MAX_LONGITUDE - self.glomn) / float(self.nlo - 1)

    def _get_latitude_index(self, latitude:float) -> Tuple[int, int]:
        """
        緯度（単位：度）から緯度インデックス値(0,1,2,.. ,self.nla -1)に変換する。

        Parameters
        ----
        latitude:float
            緯度（単位：度）

        Returns
        ----
        Tupple[int, int]
            緯度インデックスの下限値、上限値で構成されたタプル
        
        Raises
        ----
        ValueError
            緯度が範囲外の場合
        """
        if latitude < self.glamn or self.MAX_LATITUDE < latitude:
            raise ValueError(f'latitude:({latitude}) is out of range')
        lower = int(((latitude - self.glamn)/(self.MAX_LATITUDE - self.glamn))*(self.nla - 1))
        if (latitude - self.glamn) > 0.0:
            upper = lower + 1
            if self.nla <= upper:
                raise ValueError(f'upper index:({upper}) is out of range')
        else:
            upper = lower
        return (lower,upper)

    def _get_longitude_index(self, longitude:float) -> Tuple[int, int]:
        """
        経度（単位：度）から経度インデックス値(0,1,2,.. ,self.nlo -1)に変換する。

        Parameters
        ----
        longitude:float
            経度（単位：度）

        Returns
        ----
        Tupple[int, int]
            経度インデックスの下限値、上限値で構成されたタプル
        
        Raises
        ----
        ValueError
            経度が範囲外の場合
        """
        if longitude < self.glomn or self.MAX_LONGITUDE < longitude:
            raise ValueError(f'longitude:({longitude}) is out of range')
        lower = int(((longitude - self.glomn)/(self.MAX_LONGITUDE - self.glomn))*(self.nlo - 1))
        if (longitude - self.glomn) > 0.0:
            upper = lower + 1
            if self.nlo <= upper:
                raise ValueError(f'upper index:({upper}) is out of range')
        else:
            upper = lower
        return (lower,upper)

    def convert_xyz(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ジオイド高データ(self.rows)をnp.ndarray形式のリストX(緯度、単位：度)、
        Y(経度、単位：度)、Z(ジオイド高、単位：メートル)に変換する。
        ただし、ジオイド高データがない座標はリストに加えない。
        
        Returns
        ----
        Tuple[x:np.ndarray, y:np.ndarray, z:np.ndarray]
            x: 緯度、単位：度
            y: 経度、単位：度
            z: ジオイド高、単位：メートル
        """
        x = [] # 経度(東経)
        y = [] # 緯度(北緯)
        z = [] # ジオイド高(m)
        for latitude_index in range(self.nla):
            latitude = self.glamn + latitude_index * self.dgla
            for longitude_index in range(self.nlo):
                longitude = self.glomn + longitude_index * self.dglo
                if self.rows[latitude_index][longitude_index] < self.NO_DATA:
                    y.append(latitude)
                    x.append(longitude)
                    z.append(self.rows[latitude_index][longitude_index])

        if self.debug:
            print(f'x len:{len(x)}, y len:{len(y)}, z len:{len(z)}')

        # ndarray化
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        z = np.array(z, dtype=float)

        return (x, y, z)

    def get_gpd(self, crs:str='EPSG:4326') -> gpd.GeoDataFrame:
        """
        ジオイドモデルをGeoDataFrame オブジェクトとして取得する。

        Parameters
        ----
        crs:str
            座標系（デフォルト EPSG:4326）
        
        Returns
        ----
        gpd.GeoDataFrame
            ジオイドモデル(ジオイド高:'height'属性)
        """
        (x, y, z) = self.convert_xyz()
        geometry = []
        total = len(x)
        for i in range(total):
            geometry.append(Point(x[i], y[i]))
        return gpd.GeoDataFrame({'height':z, 'geometry':geometry}, crs=crs)

    @classmethod
    def to_dms(cls, degree:float) -> Tuple[int, int, float]:
        """
        度から度分秒に変換する。

        Parameters
        ----
        degree:float
            度（浮動小数点数）
        
        Returns
        ----
        Tuple[d:int, m:int, s:float]
            d: 度
            m: 分
            s: 秒（浮動小数点数）
        """
        d = int(degree)
        print(degree)
        print(d)
        print(degree - d)
        print((degree - d) * 60.0)
        m = int((degree - float(d)) * 60.0)
        print(m)
        print(degree - d - m/60.0)
        print((degree - d - m/60.0) * 3600.0)
        s = float((degree - float(float(d) + float(m) / 60.0)) * 3600.0)
        return (d, m, s)

    @classmethod
    def to_degree(cls, d:int, m:int, s:float) -> float:
        """
        度分秒から度に変換する。

        Parameters
        ----
        d:int
            度
        m:int
            分
        s:float
            秒（浮動小数点数）
        
        Returns
        ----
        degree:float
            度（浮動小数点数）
        """
        return float(float(d) + float(m)/60.0 + s/3600.0)


if __name__ == '__main__':
    """
    ジオイドモデルを変換し引数指定されたディレクトリに格納する。
    """
    import argparse
    parser = argparse.ArgumentParser(description='show Japan geoid height with 3d scatter')
    parser.add_argument('--path', type=str, default=GEOID_PATH, help='Japan Geoid Height data file(asc) path')
    parser.add_argument('--outdir', type=str, default='output', help='output directory path')
    parser.add_argument('--debug', type=bool, default=False, help='print debug lines')
    args = parser.parse_args()
    
    model = GeoidModel(path=args.path, debug=args.debug)
    model.to_all(output_dir=args.outdir)
