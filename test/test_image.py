import os, sys
sys.path.append('.')
import time
import pathlib

import cv2
import numpy as np
import torch

import config
from nets.optimized_landmark import LNet
from tools.utils import show_landmarks, show_tensor_landmarks, draw_tensor_landmarks

if __name__ == "__main__":
    model = LNet()
    # model.load_state_dict(torch.load('result/check_point/{0}_landmarks_model_200.pth'.format(config.NUM_LANDMARKS)))
    model.eval()
    model.load_state_dict(torch.load('./pretrained_weights/{0}_landmarks_model_200.pth'.format(config.NUM_LANDMARKS)))
    input_path = pathlib.Path('./data/test/')
    
    project_root = pathlib.Path()
    output_path = project_root / "result" / "imgs"
    output_path.mkdir(exist_ok=True)

    for i, input_img_filename in enumerate(input_path.iterdir()):
        img_name = input_img_filename.name
        img = cv2.imread(str(input_img_filename))
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        RGB_img = cv2.resize(RGB_img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
        
        data = RGB_img - 127.5
        data = data / 127.5
        
        data = data.transpose((2, 0, 1))
        data = np.expand_dims(data, axis=0)
        data = torch.Tensor(data)
        with torch.no_grad(): 
            landmarks = model(data)
        landmarks = landmarks.cpu().detach()

        show_tensor_landmarks(data[0], landmarks[0])
        
        landmarks = [(i) for i in landmarks[0]]
        h, w = img.shape[0:2]
        landmarks = [(landmarks[2*i]*w, landmarks[2*i+1]*h) for i in range(len(landmarks)//2)]
        show_landmarks(img, landmarks) 


