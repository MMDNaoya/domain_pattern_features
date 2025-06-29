import numpy as np
import matplotlib.pyplot as plt

# scikit-imageを中心に，OpenCVやscipy等も併用する
from skimage import measure, morphology, io
from skimage.measure import regionprops
from skimage.morphology import skeletonize, convex_hull_image
from skimage.transform import radon
from scipy.ndimage import distance_transform_edt
import cv2

import tamura


def get_subimages(arr, factor=16):
    return [
        arr[3*factor: 19*factor, 3*factor: 19*factor],
        arr[3*factor: 19*factor, 7*factor: 23*factor],
        arr[3*factor: 19*factor, 11*factor: 27*factor],
        arr[7*factor: 23*factor, 3*factor: 19*factor],
        arr[7*factor: 23*factor, 7*factor: 23*factor],
        arr[7*factor: 23*factor, 11*factor: 27*factor],
        arr[11*factor: 27*factor, 3*factor: 19*factor],
        arr[11*factor: 27*factor, 7*factor: 23*factor],
        arr[11*factor: 27*factor, 11*factor: 27*factor],
    ]


def compute_incirce_excircle_rads(region, binary_img):
    label_id = region.label
    
    # -- 領域のピクセル座標 (row, col)
    coords = region.coords
    
    # =================================================
    # 2) 内接円半径 (incircle radius) の計算
    #    領域部分だけマスクを作り、距離変換を行い、その最大値を半径とする
    # =================================================
    submask = np.zeros_like(binary_img, dtype=bool)
    submask[coords[:,0], coords[:,1]] = True  # region の画素のみ True に

    dist_map = distance_transform_edt(submask)
    incircle_radius = dist_map.max()  # 領域内で最も境界から遠い点

    # =================================================
    # 3) 外接円半径 (excircle radius) の計算
    #    OpenCV の minEnclosingCircle を使う
    #    minEnclosingCircle は (x, y) 座標で渡す必要があるので、
    #    region.coords(=row,col) -> (col,row) に変換
    # =================================================
    # OpenCV用に float32 の形 (N,1,2) に整形
    points = coords[:, ::-1].astype(np.float32).reshape(-1, 1, 2)  
    (cx, cy), excircle_radius = cv2.minEnclosingCircle(points)
    
    return incircle_radius, excircle_radius


def calculate_abstract_features(binary_img):
    """Tamuraの特徴量・Huのモーメント"""
    tamura_contrast = tamura.contrast(binary_img)
    tamura_coarseness = tamura.coarseness(binary_img)
    tamura_roughness = tamura_contrast + tamura_coarseness
    moments = cv2.moments(binary_img) 
    hu_moments = cv2.HuMoments(moments)
    
    return {
        "tamura_contrast": tamura_contrast,
        "tamura_coarseness": tamura_coarseness,
        "tamura_directionality": tamura_roughness,
        "hu0": hu_moments[0],
        "hu1": hu_moments[1],
        "hu2": hu_moments[2],
        "hu3": hu_moments[3],
        "hu4": hu_moments[4],
        "hu5": hu_moments[5],
        "hu6": hu_moments[6],
    }


def compute_geometric_features(binary_img):
    """
    2値画像(binary_img)から形状・位相的な特徴量を一括で計算し，辞書にまとめて返す関数。
    """

    binary_img = binary_img.astype(np.uint8)
    
    # --- 1. 連結成分解析 ---
    labeled_img = measure.label(binary_img, connectivity=2)
    regions = regionprops(labeled_img)
    num_components = len(regions)

    # --- 2. 各連結成分ごとの基本量を取得 ---
    # 面積，周囲長，主軸長/副軸長，アスペクト比，円形度など
    areas = []
    perimeters = []
    major_axis_lengths = []
    minor_axis_lengths = []
    aspect_ratios = []
    circularities = []
    centroids = []
    incircle_rads = []
    excircle_rads = []

    for r in regions:
        # 面積 (ピクセル数)
        area = r.area
        
        # 周囲長 (regionpropsの perimeter はガウス曲線長近似; cv2等で輪郭から厳密値も可)
        perimeter = r.perimeter
        
        # 主軸長・副軸長 (ellipse fit相当)
        major_axis = r.major_axis_length
        minor_axis = r.minor_axis_length
        
        # アスペクト比
        aspect_ratio = (major_axis / minor_axis) if minor_axis != 0 else np.nan
        
        # 円形度: (4π × 面積) / (周囲長^2)
        # 周囲長が0の特殊ケースに注意
        circularity = 4.0 * np.pi * area / (perimeter**2) if perimeter > 0 else np.nan
        
        # セントロイド (重心)
        centroid = r.centroid

        # 内接円・外接円
        incircle_rad, excircle_rad = compute_incirce_excircle_rads(r, binary_img)
        
        areas.append(area)
        perimeters.append(perimeter)
        major_axis_lengths.append(major_axis)
        minor_axis_lengths.append(minor_axis)
        aspect_ratios.append(aspect_ratio)
        circularities.append(circularity)
        incircle_rads.append(incircle_rad)
        excircle_rads.append(excircle_rad)
        centroids.append(centroid)

    # --- 3. 画像全体のMinkowski functionals (2D) ---
    #  ・面積: True画素総数
    #  ・周囲長(界面長)
    #  ・オイラー数(Euler characteristic): connected_components - holes
    # scikit-imageの measure.euler の第2引数は connectivity
    euler_number = measure.euler_number(binary_img, connectivity=1)
    
    total_area = np.sum(binary_img)
    total_perimeter = measure.perimeter_crofton(binary_img)
    
    # --- 4. スケルトン解析に基づく指標 ---
    skeleton = skeletonize(binary_img)
    # スケルトンのピクセル数(=骨格の総長に近似)を計算
    skeleton_length = np.sum(skeleton)

    # --- 5. 最近傍粒子間距離の分布 ---
    #   セントロイド同士の最短距離を計算
    #   粒子が1つ以下なら計算不可なのでエラー回避
    nearest_neighbor_distances = []
    if num_components > 1:
        centroids_arr = np.array(centroids)
        # 全ペア距離を求めたうえで, それぞれ最近傍を取る
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(centroids_arr, centroids_arr)
        # 対角成分(=0)を除外する
        dist_matrix[dist_matrix == 0] = np.inf
        nn_dist = np.min(dist_matrix, axis=1)
        nearest_neighbor_distances = nn_dist
    else:
        nearest_neighbor_distances = np.array([])

    # --- 6. 凸包関連の指標 (例：凸包面積比) ---
    #   全体のバイナリマスクに対して凸包を取る (各粒子に対して個別に行うこともある)
    chull = convex_hull_image(binary_img)
    chull_area = np.sum(chull)
    # 凸包面積比（実際の面積 / 凸包面積）
    #  - 連結領域が多数ある場合は粒子毎にやる方が物理的には妥当
    if chull_area > 0:
        convexity_ratio = total_area / chull_area
    else:
        convexity_ratio = np.nan

    # --- 7. ---
    abstract_features = calculate_abstract_features(binary_img)

    # 結果をまとめる (大域統計量 + 各粒子ベース)
    results = {
        "num_components": num_components,
        "areas": areas,
        "perimeters": perimeters,
        "major_axis_lengths": major_axis_lengths,
        "minor_axis_lengths": minor_axis_lengths,
        "aspect_ratios": aspect_ratios,
        "circularities": circularities,
        "centroids": centroids,
        "incircle_rads": incircle_rads,
        "excircle_rads": excircle_rads,
        
        # Minkowski functionals (2D) の一例
        "total_area": total_area,
        "total_perimeter": total_perimeter,
        "euler_number": euler_number,
        
        # スケルトン情報
        "skeleton_length": skeleton_length,
        
        # 最近傍距離分布
        "nearest_neighbor_distances": nearest_neighbor_distances,

        # 凸包
        "convex_hull_area": chull_area,
        "convexity_ratio": convexity_ratio,}
    results.update(abstract_features)
    return results


def calculate_all_features(arr):
    """arrは+-1のアレイ．+1を図，-1を地とする統計量と，逆に-1を図，+1を地とする特徴量の両方を得る．"""
    features_pos = compute_geometric_features(arr > 0)
    features_neg = compute_geometric_features(arr <= 0)
    result = {"pos_"+k: v for k, v in features_pos.items()}
    result.update({"neg_"+k: v for k, v in features_neg.items()})
    return result
