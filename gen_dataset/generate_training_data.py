import os, sys

import numpy as np


def normalize_landmark(landmark, rect):
    x, y, w, h = rect
    normalize = [0] * len(landmark)
    for i in range(len(landmark)//2):
        normalize[2*i] = (landmark[2*i] - x) / w
        normalize[2*i+1] = (landmark[2*i+1] - y) / h
    return normalize


anno_file = open('annotations/iris_landmark.txt', 'w')

with open('annotations/eye_bbox_landmark.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        print(line)
        line = line.strip()
        line = line.split(' ')
        img_name = line[0]
        bbox =  [int(i) for i in line[1:5]]
        landmark =[float(i) for i in line[5:]]
        normalized_landmark = normalize_landmark(landmark, bbox)
        normalized_landmark = [str(i) for i in normalized_landmark]
        anno_file.write(img_name+' ')
        anno_file.write(' '.join(normalized_landmark))
        anno_file.write('\n')
