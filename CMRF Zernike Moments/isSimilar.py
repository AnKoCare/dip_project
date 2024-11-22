import numpy as np

def isSimilar(fv1: np.ndarray, fv2: np.ndarray, threshold: float) -> bool:
    """
    Kiểm tra độ tương đồng giữa hai vector đặc trưng
    
    Parameters:
        fv1: Vector đặc trưng thứ nhất
        fv2: Vector đặc trưng thứ hai
        threshold: Ngưỡng tương đồng
        
    Returns:
        bool: True nếu hai vector đủ tương đồng, False nếu ngược lại
    """
    return np.linalg.norm(fv1 - fv2) <= threshold