import numpy as np
from isSimilar import isSimilar
def getMatches(feature_matrix, locations, search_th, threshold, distance_th, 
               size_matchlist=None, start=0):
    """
    Tìm các cặp block tương đồng trong ảnh
    
    Parameters:
        feature_matrix: ma trận đặc trưng của các block
        locations: vị trí các block
        search_th: ngưỡng tìm kiếm
        threshold: ngưỡng tương đồng
        distance_th: ngưỡng khoảng cách
        size_matchlist: kích thước danh sách kết quả
        start: vị trí bắt đầu (default: 0)
    
    Returns:
        match_list: danh sách các cặp block tương đồng
        num: số lượng cặp tìm thấy
    """
    # Khởi tạo match_list
    if size_matchlist is None:
        match_list = []
    else:
        match_list = np.zeros((size_matchlist, 2), dtype=int)
    
    # Lấy kích thước ma trận đặc trưng
    size_featurematrix = feature_matrix.shape[0]
    num = 0
    
    # Tìm các cặp block tương đồng
    for u in range(size_featurematrix - search_th):
        for v in range(search_th):
            # Tính khoảng cách giữa hai block
            if np.linalg.norm(locations[u] - locations[u+v]) > distance_th:
                # Kiểm tra độ tương đồng
                if isSimilar(feature_matrix[u], feature_matrix[u+v], threshold):
                    if size_matchlist is None:
                        # Thêm vào list động
                        match_list.append([start+u, start+u+v])
                    else:
                        # Thêm vào mảng cố định
                        match_list[num] = [start+u, start+u+v]
                    num += 1
    
    # Chuyển list sang numpy array nếu cần
    if size_matchlist is None:
        match_list = np.array(match_list)
    
    return match_list, num