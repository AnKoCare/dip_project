import numpy as np
from Zernikmoment import Zernikmoment

def getFeatureVector(block, FVsize, Vstars):
    # Khởi tạo vector đặc trưng
    FeatureVector = np.zeros(13)  # 13 vì có thêm chỉ số 13 ở dòng 2
    
    # Tính các Zernike moments
    Amp, Phi = Zernikmoment(block, Vstars[:,:,0], 0)
    FeatureVector[0] = Amp
    
    # Trường hợp đặc biệt trả về 2 giá trị
    Amp, Phi = Zernikmoment(block, Vstars[:,:,1], 1)
    FeatureVector[1] = Amp
    FeatureVector[12] = Phi  # Lưu giá trị Phi vào vị trí 12
    
    # Các moments còn lại
    moments_config = [
        (2, 2, 2), (3, 3, 2), (4, 4, 3), (5, 5, 3),
        (6, 6, 4), (7, 7, 4), (8, 8, 4), (9, 9, 5),
        (10, 10, 5), (11, 11, 5)
    ]
    
    for idx, vstar_idx, n in moments_config:
        Amp, Phi = Zernikmoment(block, Vstars[:,:,vstar_idx], n)
        FeatureVector[idx] = Amp  # Chỉ lấy giá trị Amp
    
    # Trả về chỉ FVsize phần tử đầu tiên
    return FeatureVector[:FVsize]