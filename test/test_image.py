import os
import sys
import time
import pathlib
import cv2
import numpy as np
import torch
from nets.optimized_landmark import LNet
from tools.utils import show_landmarks, show_tensor_landmarks

if __name__ == "__main__":
    model = LNet()
    pretrained_weights_path = './pretrained_weights/{0}_landmarks_model_200.pth'.format(config.NUM_LANDMARKS)

    if os.path.exists(pretrained_weights_path):
        model.load_state_dict(torch.load(pretrained_weights_path))
        print("Loaded pretrained weights successfully.")
    else:
        print("Pretrained weights not found. Please check the path:", pretrained_weights_path)
        sys.exit(1)

    model.eval()

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
            
        landmarks = landmarks.cpu().detach().numpy()[0]

        # Visualize landmarks using the show_tensor_landmarks function
        show_tensor_landmarks(data[0], landmarks)

        # Convert normalized landmarks to image coordinates
        h, w = img.shape[0:2]
        landmarks_img_coords = [(int(landmarks[2*i] * w), int(landmarks[2*i+1] * h)) for i in range(len(landmarks)//2)]

        # Visualize landmarks using the show_landmarks function
        show_landmarks(img, landmarks_img_coords)
