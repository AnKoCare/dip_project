import numpy as np
from radialpoly import radialpoly

def getVstar(siz, n, m):
    """
    Calculate V* for Zernike moments
    
    Parameters:
    siz (int): Size of the output matrix
    n (int): Radial polynomial degree
    m (int): Angular frequency
    
    Returns:
    numpy.ndarray: Complex matrix V*
    """
    N = siz
    x = np.arange(1, N+1)
    y = x.copy()
    
    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    
    # Calculate radius and angle
    R = np.sqrt((2*X - N - 1)**2 + (2*Y - N - 1)**2) / N
    Theta = np.arctan2((N-1-2*Y+2), (2*X-N+1-2))
    
    # Create mask for unit circle
    Mask = R <= 1
    
    # Calculate V* using radial polynomial
    V = Mask * radialpoly(R, n, m) * np.exp(-1j * m * Theta)
    
    return V