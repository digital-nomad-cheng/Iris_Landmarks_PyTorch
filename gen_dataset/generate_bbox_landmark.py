import os, sys
import glob
import json

import cv2
import numpy as np

DISPLAY = 0


def enlarge_eye(img, rect, ratio):
    if DISPLAY:
        cv2.imshow('before', img[rect[1]: rect[1]+rect[3], 
                                rect[0]: rect[0]+rect[2]])
        cv2.waitKey(0)
    img_h, img_w = img.shape[0:2]
    w, h = rect[2], rect[3]
    dw = int(w*ratio)
    dh = int(h*ratio)

    larger_x = 0 if rect[0] - dw < 0 else rect[0] - dw
    larger_y = 0 if rect[1] - dh < 0 else rect[1] - dh
    larger_xx = rect[0] + rect[2] + dw if rect[0] + rect[2] + dw < img_w else img_w
    larger_yy = rect[1] + rect[3] + dh if rect[1] + rect[3] + dh < img_h else img_h
    
    larger_w = larger_xx - larger_x
    larger_h = larger_yy - larger_y

    larger_rect = [larger_x, larger_y, larger_w, larger_h]

    larger_eye = img[larger_rect[1]: larger_rect[1]+larger_rect[3], 
                     larger_rect[0]: larger_rect[0]+larger_rect[2]]
    
    if DISPLAY:
        cv2.imshow('after', larger_eye)
        cv2.waitKey(0)
    return larger_eye, larger_rect

def generate_eye_img_and_landmark(json_file):

    img_file = "%s.jpg" % json_file[:-5]
    print(img_file)
    org_img = cv2.imread(img_file)
    img = org_img.copy()
    data_file = open(json_file)
    ldmks = json.load(data_file)
     
    def process_json_list(json_list):
        ldmks = [eval(s) for s in json_list]
        return np.array([(x, img.shape[0]-y, z) for (x, y, z) in ldmks])
    
    interior_margin_ldmks = process_json_list(ldmks['interior_margin_2d'])
    caruncle_ldmks = process_json_list(ldmks['caruncle_2d'])
    iris_ldmks = process_json_list(ldmks['iris_2d'])
    
    if DISPLAY:
        for ldmk in interior_margin_ldmks:
            cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 2, (0, 255, 0), -1)
        cv2.imshow('interiro_margin', img)
        cv2.waitKey(0)
        for ldmk in caruncle_ldmks:
            cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 2, (0, 255, 0), -1)
        cv2.imshow('caruncle', img)
        cv2.waitKey(0)
        for ldmk in iris_ldmks:
            cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 2, (0, 255, 0), -1)
        cv2.imshow('iris', img)
        cv2.waitKey(0)

    ldmks = np.vstack([interior_margin_ldmks, caruncle_ldmks, iris_ldmks])  
    ldmks = np.expand_dims(ldmks, axis=1).astype('float32')
    
    rect = cv2.boundingRect(ldmks[:, :, 0:2])
    
    if DISPLAY:
        cv2.rectangle(img, rect, (255, 0, 0), 1)
        cv2.imshow('rect', img)
        cv2.waitKey(0)
    
    
    larger_eye, larger_rect = enlarge_eye(org_img, rect, 0.25)
     
    return larger_eye, larger_rect, iris_ldmks 
    

if __name__ == "__main__":
    DISPLAY = 0
    data_path = '/home/idealabs/data/opensource_dataset/Iris_landmark'
    training_data_path = 'data/train'
    if not os.path.exists(training_data_path):
        os.makedirs(training_data_path)

    f = open('annotations/eye_bbox_landmark.txt', 'w')
    json_files = glob.glob(os.path.join(data_path, '*.json'))
    for json_file in json_files:
        print(json_file)
        json_name = os.path.basename(json_file)
        img_name = '%s.jpg' % json_name[:-5]
        eye, rect, iris_ldmks = generate_eye_img_and_landmark(json_file)
        cv2.imwrite(os.path.join(training_data_path, img_name), eye)
        f.write(img_name+' ')
        rect = [str(i) for i in rect]
        f.write(' '.join(rect) + ' ')
        for i in range(iris_ldmks.shape[0]):
            f.write(str(iris_ldmks[i][0]) + ' ')
            f.write(str(iris_ldmks[i][1]) + ' ')
        f.write('\n')
