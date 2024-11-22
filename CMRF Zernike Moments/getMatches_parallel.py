import numpy as np
from multiprocessing import Pool
from typing import Tuple, List
from getMatches import getMatches

# Move 'process_section' to the top level
def process_section(args):
    idx, feat, loc, search_th, threshold, distance_th, size_matchlist, section_size = args
    start_idx = idx * (section_size - search_th)
    return getMatches(feat, loc, search_th, threshold, distance_th, size_matchlist, start_idx)

def getMatches_parallel(feature_matrix: np.ndarray, 
                       locations: np.ndarray,
                       threshold: float,
                       search_th: int,
                       distance_th: float) -> np.ndarray:
    """
    Tìm các cặp block tương đồng song song trên 4 CPU core
    
    Parameters:
        feature_matrix: Ma trận đặc trưng của các block
        locations: Vị trí các block
        threshold: Ngưỡng tương đồng
        search_th: Ngưỡng tìm kiếm
        distance_th: Ngưỡng khoảng cách
    """
    num_blocks, fv_size = feature_matrix.shape
    
    # Mở rộng ma trận với các hàng vô cùng
    feature_matrix = np.vstack([
        feature_matrix,
        np.full((search_th, fv_size), np.inf)
    ])
    
    locations = np.vstack([
        locations,
        np.full((search_th, 2), np.inf)
    ])
    
    # Chia thành 4 phần cho xử lý song song
    section_size = num_blocks // 4 + search_th
    sub_features = np.zeros((4, section_size, fv_size))
    sub_locations = np.zeros((4, section_size, 2))
    
    # Phân chia dữ liệu
    for k in range(4):
        start = k * (section_size - search_th)
        end = start + section_size
        sub_features[k] = feature_matrix[start:end]
        sub_locations[k] = locations[start:end]
    
    # Kích thước tối đa cho danh sách kết quả
    size_matchlist = (section_size - search_th) * search_th
    
    # Xử lý song song
    with Pool(processes=4) as pool:
        results = pool.map(process_section, [
            (i, sub_features[i], sub_locations[i],
             search_th, threshold, distance_th, size_matchlist, section_size)
            for i in range(4)
        ])
    
    # Kết hợp kết quả
    match_lists, nums = zip(*results)
    result = np.vstack([
        match_lists[i][:nums[i]] for i in range(4)
    ])
    
    return result