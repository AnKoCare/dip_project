import numpy as np

def getCircleMask(B: int) -> np.ndarray:
    """
    Tạo mask hình tròn với kích thước BxB
    
    Parameters:
        B: Kích thước cạnh của mask vuông
    
    Returns:
        np.ndarray: Mask boolean hình tròn
    """
    # Tạo lưới tọa độ
    y, x = np.ogrid[:B, :B]
    
    # Tính tâm
    center = (B - 1) / 2
    
    # Tính khoảng cách từ mỗi điểm đến tâm
    distances = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Tạo mask với bán kính B/2
    circle = distances <= B/2
    
    return circle