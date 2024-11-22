import numpy as np
from getFeatureVector import getFeatureVector
from getVstar import getVstar
def getFeatureMatrix(grayimage, B, FVsize):
    # Khởi tạo ma trận đặc trưng
    Vstars = np.zeros((B, B, 12), dtype=complex)
    
    # Tính các V-stars
    n_m_pairs = [
        (0,0), (1,1), (2,0), (2,2), (3,1), (3,3),
        (4,0), (4,2), (4,4), (5,1), (5,3), (5,5)
    ]
    
    for idx, (n, m) in enumerate(n_m_pairs):
        Vstars[:,:,idx] = getVstar(B, n, m)
    
    # Lấy kích thước ảnh
    M, N = grayimage.shape
    num_blocks = (M-B+1) * (N-B+1)  # số lượng block
    
    # Khởi tạo ma trận kết quả
    FeatureMatrix = np.zeros((num_blocks, FVsize))
    Locations = np.zeros((num_blocks, 2))
    
    # Trích xuất đặc trưng từ mỗi block
    rownum = 0
    for x in range(N-B+1):
        for y in range(M-B+1):
            # Lấy block
            block = grayimage[y:y+B, x:x+B]
            
            # Lưu đặc trưng
            FeatureMatrix[rownum,:] = getFeatureVector(block, FVsize, Vstars)
            Locations[rownum,:] = [x, y]
            
            rownum += 1
            
    return FeatureMatrix, Locations