import os

import cv2
import numpy as np


landmarks_path = "./data/test/result.txt"
data_path = "./data/test"

left_eye_index = [67, 68, 69, 70, 71, 72, 73, 74]
right_eye_index = [76, 77, 78, 79, 80, 81, 82, 83]



def get_larger_rect(rect, img_h, img_w, ratio=0.15):
    w, h = rect[2], rect[3]
    dw = int(w * ratio)
    dh = int(h * ratio)

    larger_rect = [0] * 4
    larger_rect[0] = (rect[0] - dw) if (rect[0] - dw) > 0 else 0
    larger_rect[1] = (rect[1] - dw) if (rect[1] - dh) > 0 else 0
    larger_rect[2] = (rect[2] + 2*dw) if (rect[2] + 2*dw) < img_w else img_w
    larger_rect[3] = (rect[3] + 2*dw) if (rect[3] + 2*dh) < img_h else img_h
    
    return larger_rect

def crop_eye_with_landmarks(img, landmarks):
    left_eye_points = []
    right_eye_points = []
    
    print(len(landmarks))
    for i in left_eye_index:
        i -= 1
        left_eye_points.append((landmarks[2*i], landmarks[2*i+1]))

    for i in right_eye_index:
        i -= 1
        right_eye_points.append((landmarks[2*i], landmarks[2*i+1]))

    left_eye_rect = cv2.boundingRect(np.array(left_eye_points))
    right_eye_rect = cv2.boundingRect(np.array(right_eye_points))
    
    h, w = img.shape[0:2]
    larger_left_eye_rect = get_larger_rect(left_eye_rect, h, w)
    larger_right_eye_rect = get_larger_rect(right_eye_rect, h, w)
    
    large_rect = larger_left_eye_rect
    left_eye = img[large_rect[1]:large_rect[1]+large_rect[3],
               large_rect[0]:large_rect[0]+large_rect[2]]
    larger_rect = larger_right_eye_rect
    right_eye = img[large_rect[1]:large_rect[1]+large_rect[3],
               large_rect[0]:large_rect[0]+large_rect[2]]
    
    return left_eye, right_eye

with open(landmarks_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        items = line.split(' ')
        img_name = items[0]
        landmarks = [int(i) for i in items[5:]]
        if len(landmarks) == 0:
            continue
        img = cv2.imread(os.path.join(data_path, img_name), 1)

        left_eye, right_eye = crop_eye_with_landmarks(img, landmarks)
        
        cv2.imwrite(os.path.join(data_path, 'left_'+img_name), left_eye)
        cv2.imwrite(os.path.join(data_path, 'right_'+img_name), right_eye)



