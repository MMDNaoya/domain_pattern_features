import numpy as np
import features_calculator


def argmedian(arr):
    median_value = np.median(arr)
    return np.argmin(np.abs(arr - median_value))

def get_feats_by_area(features, mode='pos'):
    """
    各種特徴量の総和と，面積を基準として最大/中央値/最小の領域の各種特徴量を取得する。
    ただし対象となる領域が1つもない場合は、各特徴量を None で埋める。総和(total)は0とする．
    """
    feat_names = [
        'area', 'perimeter', 'major_axis_length', 'minor_axis_length',
        'aspect_ratio', 'circularitie', 'nearest_neighbor_distance',
        'incircle_rad', 'excircle_rad'
    ]
    
    # 領域が1つもない場合の例外処理
    if len(features[mode+'_areas']) == 0:
        res = {}
        # 「最大」「最小」「中央値」の各接頭辞を付けて全て None
        for feat_name in feat_names:
            res[mode + "_largest_byarea_" + feat_name] = None
            res[mode + "_smallest_byarea_" + feat_name] = None
            res[mode + "_median_byarea_" + feat_name] = None
            res[mode + "_total_byarea_" + feat_name] = 0 # 空集合の総和は・・・とりあえず0で
        return res
    if len(features[mode+'_areas']) == 1:
        features[mode + "_nearest_neighbor_distances"] = [None]
    
    # 通常通り計算
    largest_area_index = np.argmax(features[mode+'_areas'])
    smallest_area_index = np.argmin(features[mode+'_areas'])
    median_area_index = argmedian(features[mode+'_areas'])

    largest_feats = {
        mode + "_largest_byarea_" + feat_name:
            features[mode + '_' + feat_name + 's'][largest_area_index]
        for feat_name in feat_names
    }
    smallest_feats = {
        mode + "_smallest_byarea_" + feat_name:
            features[mode + '_' + feat_name + 's'][smallest_area_index]
        for feat_name in feat_names
    }
    median_feats = {
        mode + "_median_byarea_" + feat_name:
            features[mode + '_' + feat_name + 's'][median_area_index]
        for feat_name in feat_names
    }
    total_feats = {
        mode + "_total_byarea_" + feat_name:
            np.sum(features[mode + '_' + feat_name + 's'])
        for feat_name in feat_names
    }

    res = {}
    res.update(largest_feats)
    res.update(smallest_feats)
    res.update(median_feats)
    res.update(total_feats)
    return res

def get_stats_list(features, mode='pos'):
    """
    リスト形式（各領域ごとの値）で与えられた特徴量の最大値・中央値・最小値・総和を取得。
    ただし対象となる領域が1つもない場合は、各特徴量を None で埋める。
    """
    feat_names = ['perimeter', 'major_axis_length', 'minor_axis_length',
                  'aspect_ratio', 'circularitie']
    
    # 領域が1つもない場合の例外処理
    if len(features[mode+'_areas']) == 0:
        res = {}
        for feat_name in feat_names:
            res[mode + "_max_" + feat_name] = None
            res[mode + "_median_" + feat_name] = None
            res[mode + "_min_" + feat_name] = None
            res[mode + "_total_" + feat_name] = None
        return res

    # 通常通り計算
    dictionaries = []
    for feat_name in feat_names:
        data_array = features[mode + '_' + feat_name + 's']
        dictionaries.append({
            mode + "_max_" + feat_name: np.max(data_array),
            mode + "_median_" + feat_name: np.median(data_array),
            mode + "_min_" + feat_name: np.min(data_array),
            mode + "_total_" + feat_name: np.sum(data_array)
        })

    result = {}
    for d in dictionaries:
        result.update(d)
    return result

def get_single_value(features, mode='pos'):
    """
    領域数や合計面積など、単一値として保持される特徴量をまとめて取得。
    領域が1つもない場合は None にする。
    """
    feat_names = [
        'num_components', 'total_area', 'total_perimeter',
        'euler_number', 'skeleton_length', 'convex_hull_area',
        'convexity_ratio',
        'tamura_contrast', 'tamura_coarseness', 'tamura_directionality',
        'hu0', 'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6'
    ]
    # 「領域数 (num_components) が0」なら他の単一値も全て None とする
    if features[mode + '_num_components'] == 0:
        return {mode + '_' + fn: None for fn in feat_names}

    return {mode + '_' + fn: features[mode + '_' + fn] for fn in feat_names}

def get_all_features(features):
    """
    pos/neg それぞれのモードに対して、
    領域ベースの特徴量と単一値の特徴量をまとめて取得。
    """
    result = {}
    for mode in ('pos', 'neg'):
        # 面積を基準にした最大/最小/中央値
        res_area = get_feats_by_area(features, mode)
        result.update(res_area)

        # リスト系特徴量の統計量
        res_stats = get_stats_list(features, mode)
        result.update(res_stats)

        # 単一値の特徴量
        res_single = get_single_value(features, mode)
        result.update(res_single)

    return result