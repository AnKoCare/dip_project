import numpy as np

def Zernikmoment(p, Vstar, n):
    """
    Calculate Zernike moments for an image
    
    Parameters:
        p (ndarray): Input image
        Vstar (ndarray): Complex conjugate of Zernike polynomial
        n (int): Order of Zernike polynomial
    
    Returns:
        tuple: (Amp, Phi) where:
            Amp (float): Amplitude of the moment
            Phi (float): Phase of the moment in degrees
    """
    # Calculate product and sum
    Product = p * Vstar
    Z = np.sum(Product)
    
    # Normalize the amplitude
    Z = (n + 1) * Z / np.pi
    
    # Calculate amplitude and phase
    Amp = np.abs(Z)
    Phi = np.angle(Z) * 180 / np.pi
    
    return Amp, Phi