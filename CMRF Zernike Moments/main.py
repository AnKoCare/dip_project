import cv2
import numpy as np
import time
from pathlib import Path
import sys
from getFeatureMatrix_parallel import getFeatureMatrix_parallel
from getMatches_parallel import getMatches_parallel
from showMatches import showMatches

def copy_move_detection():
    """Main function for Copy-Move-Forgery detection"""
    
    # Initialize parameters
    method = ''
    filename = input('Enter File Name: ')
    result_name = str(Path('results') / f"{method}{filename}")
    color = [0, 0, 255]
    
    # Load and convert image
    rgb_image = cv2.imread(filename)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY).astype(float)
    
    # Get image dimensions
    M, N = gray_image.shape
    
    # Set detection parameters
    B = 24  # Block size
    search_th = 50  # Search threshold
    distance_th = 50  # Distance threshold
    FVsize = 12  # Feature vector length
    similarity_threshold = 0.1  # Similarity factor
    
    # Start timing
    start_time = time.time()
    
    # Extract features from blocks
    feature_matrix, locations = getFeatureMatrix_parallel(gray_image, B, FVsize)
    
    # Sort features
    sorted_indices = np.argsort(feature_matrix[:,0])
    feature_matrix = feature_matrix[sorted_indices]
    locations = locations[sorted_indices]
    
    # Find matches
    match_list = getMatches_parallel(feature_matrix, locations, 
                                   similarity_threshold, search_th, distance_th)
    
    # Show results
    showMatches(rgb_image, match_list, locations, B, color, 1, 1, result_name)
    
    # Print elapsed time
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    copy_move_detection()