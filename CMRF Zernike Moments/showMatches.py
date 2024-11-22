import cv2
import numpy as np
from pathlib import Path
from getCircleMask import getCircleMask
def showMatches(RGB_image, match_list, locations, B, color, show_map=True, 
                show_org=True, result_name=None):
    """
    Hiển thị kết quả phát hiện copy-move
    
    Parameters:
        RGB_image: ảnh gốc
        match_list: danh sách các cặp block tương đồng
        locations: vị trí các block
        B: kích thước block
        color: màu đánh dấu [B,G,R]
        show_map: hiển thị bản đồ copy-move
        show_org: hiển thị ảnh gốc
        result_name: tên file kết quả
    """
    # Tạo mask hình tròn
    mask = getCircleMask(B)
    
    # Lấy kích thước ảnh
    M, N = RGB_image.shape[:2]
    
    # Tạo bản đồ nhị phân
    map_binary = np.zeros((M, N), dtype=bool)
    
    # Đánh dấu các vùng copy-move
    if len(match_list) > 0:
        for match in match_list:
            u, v = match
            # Chuyển từ 1-based sang 0-based indexing
            x1, y1 = int(locations[u-1,0]-1), int(locations[u-1,1]-1)
            x2, y2 = int(locations[v-1,0]-1), int(locations[v-1,1]-1)
            
            # Cập nhật map với mask
            map_binary[y1:y1+B, x1:x1+B] |= mask
            map_binary[y2:y2+B, x2:x2+B] |= mask
    
    # Tạo ảnh kết quả cuối cùng
    ultimate_result = RGB_image.copy()
    
    # Đánh dấu vùng copy-move bằng màu đã chọn
    ultimate_result[map_binary] = color
    
    # Hiển thị kết quả
    if show_org:
        cv2.imshow('Original', RGB_image)
        
    if show_map:
        cv2.imshow('Copy-Move Map', map_binary.astype(np.uint8) * 255)
        
    cv2.imshow(str(result_name), ultimate_result)
    
    # Lưu kết quả nếu có tên file
    if result_name:
        output_path = Path(result_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), ultimate_result)
    
    # Đợi phím bất kỳ để đóng cửa sổ
    cv2.waitKey(0)
    cv2.destroyAllWindows()