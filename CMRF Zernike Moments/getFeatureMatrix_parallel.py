import numpy as np
from multiprocessing import Pool
from getFeatureMatrix import getFeatureMatrix

def process_subimage(args):
    sub_img, B, FVsize = args
    return getFeatureMatrix(sub_img, B, FVsize)

def getFeatureMatrix_parallel(grayimage, B, FVsize):
    # Get image size
    M, N = grayimage.shape
    
    # Crop image for parallel computing
    if (M + B) % 2 == 0:
        M = M - 1
        grayimage = grayimage[:M, :]
    if (N + B) % 2 == 0:
        N = N - 1
        grayimage = grayimage[:, :N]
    
    # Calculate center points
    c1 = (M + B - 1) // 2
    c2 = (N + B - 1) // 2
    
    # Create sub images array
    sub_images = np.zeros((4, c1, c2))
    sub_images[0] = grayimage[:c1, :c2]
    sub_images[1] = grayimage[:c1, (c2-B+1):N]
    sub_images[2] = grayimage[(c1-B+1):M, :c2]
    sub_images[3] = grayimage[(c1-B+1):M, (c2-B+1):N]
    
    # Define sections
    sections = np.array([
        [0, 0],
        [0, c2-B+1],
        [c1-B+1, 0],
        [c1-B+1, c2-B+1]
    ])
    
    # Prepare arrays for results
    num_blocks = (c1-B+1)*(c2-B+1)
    Mats = np.zeros((4, num_blocks, FVsize))
    locs = np.zeros((4, num_blocks, 2))
    tasks = [(sub_images[i], B, FVsize) for i in range(4)]
    # Parallel processing using multiprocessing
    # def process_subimage(args):
    #     h, sub_img = args
    #     return getFeatureMatrix(sub_img, B, FVsize)
    
    with Pool(processes=4) as pool:
       results = pool.map(process_subimage, tasks)
    
    # Unpack results
    for i, (mat, loc) in enumerate(results):
        Mats[i] = mat
        locs[i] = loc
    
    # Combine results
    FeatureMatrix = np.vstack([Mats[i] for i in range(4)])
    
    # Combine locations with offsets
    Locations = np.vstack([
        np.column_stack((locs[i,:,0] + sections[i,1], 
                        locs[i,:,1] + sections[i,0])) 
        for i in range(4)
    ])
    
    return FeatureMatrix, Locations