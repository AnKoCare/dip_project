import numpy as np
from math import factorial

def radialpoly(r, n, m):
    """
    Compute Zernike Polynomials
    
    Parameters:
        r (ndarray): radius
        n (int): order of Zernike polynomial
        m (int): repetition of Zernike moment
    
    Returns:
        ndarray: Radial polynomial values
    """
    # Khởi tạo mảng kết quả với cùng kích thước như r
    rad = np.zeros_like(r)
    
    # Tính toán đa thức Zernike
    for s in range(0, int((n-abs(m))//2 + 1)):
        c = ((-1)**s * factorial(n-s) / 
             (factorial(s) * 
              factorial(int((n+abs(m))/2)-s) * 
              factorial(int((n-abs(m))/2)-s)))
        rad = rad + c * r**(n-2*s)
    
    return rad