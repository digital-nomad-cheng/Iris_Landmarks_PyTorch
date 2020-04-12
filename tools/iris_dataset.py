import os, sys
sys.path.append('.')

import random

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

import config
from tools.utils import show_tensor_landmarks

def normalize_landmark(landmark, image):
    h, w = image.shape[0:2]
    for i in range(landmark.shape[0]):
        landmark[i, 0] /= w
        landmark[i, 1] /= h
    return landmark

def unnormalize_landmark(landmark, image):
    h, w = image.shape[0:2]
    for i in range(landmark.shape[0]):
        landmark[i, 0] *= w
        landmark[i, 1] *= h
    return landmark

class Resize(object):
    """Rescale the image in a sample to a given size.
    args:
        output_size: int or tuple
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_w, new_h = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        # landmark is already normalized 
        return {'image':img, 'landmarks': landmarks}

class RandomFlip(object):
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, sample):
        image = sample['image']
        landmarks = sample['landmarks']
        if random.random() < self.prob:
            image = cv2.flip(image, 1) # 1 for flip around y axis, 0 for x axis, -1 for both
            # landmarks[:,0] = image.shape[1] - landmarks[:,0] # flip x coordinates
            landmarks[:, 0] = 1 - landmarks[:, 0]
        return {'image': image, 'landmarks': landmarks}

class Rescale(object):
    """Downscale and upscale an image"""
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, sample):
        ratio = np.random.randint(self.low, self.high)
        image = sample['image']
        landmarks = sample['landmarks']
        h, w = image.shape[0:2]
        image = cv2.resize(image, (int(w/ratio), int(h/ratio)))
        image = cv2.resize(image, (w, h))
        return {'image': image, 'landmarks': landmarks}

class RandomGaussianBlur(object):
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, sample):
        image = sample['image']
        if random.random() < self.prob:
            image = cv2.GaussianBlur(image, (11, 11), 0)
        sample['image'] = image
        return sample

class RandomMedianBlur(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample):
        image = sample['image']
        if random.random() < self.prob:
            image = cv2.medianBlur(image, 11)

        sample['image'] = image
        return sample

class RandomCropResize(object):
    def __init__(self, output_size, resize_ratio):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.resize_ratio = resize_ratio

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        resize = np.random.random()
        
        h, w = image.shape[:2]
        new_w, new_h = self.output_size
        
        if resize < self.resize_ratio:

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            landmarks = unnormalize_landmark(landmarks, image)
            image = image[top:top + new_h,
                      left:left+new_w]
            landmarks = landmarks - [left, top]
            landmarks = normalize_landmark(landmarks, image)
        else:
            image = cv2.resize(image, (new_w, new_h))
            # landmarks = landmarks * [new_w / w, new_h / h]
        return {'image': image, "landmarks": landmarks}

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree
    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[0:2]
        landmarks = sample['landmarks']
        img_h, img_w = image.shape[0:2]
        center = (img_w // 2, img_h // 2)
        random_degree = np.random.uniform(-self.degree, self.degree)
        rot_mat = cv2.getRotationMatrix2D(center, random_degree, 1)
        image_rotated = cv2.warpAffine(image, rot_mat, (img_w, img_h))

        landmark_rotated = np.asarray([(rot_mat[0][0]*x*w+rot_mat[0][1]*y*h+rot_mat[0][2], 
                                        rot_mat[1][0]*x*w+rot_mat[1][1]*y*h+rot_mat[1][2]) 
                                        for (x, y) in landmarks])
        
        for i in range(landmark_rotated.shape[0]//2):
            landmark_rotated[2*i] /= w
            landmark_rotated[2*i+1] /= 2

        return {'image': image_rotated, "landmarks": landmark_rotated}

class RandomMotionBlur(object):
    def __init__(self, radius):
        self.radius = radius
        self.seq = iaa.Sequential([
            iaa.Sometimes(0.2, 
                iaa.MotionBlur(k=self.radius)
            )
        ])
    def __call__(self, sample):
        image = sample['image']
        landmarks = sample['landmarks']
        landmarks = unnormalize_landmark(landmarks, image) 
        print(landmarks)
        kps = self.landmarks_to_kps(image, landmarks) 
        img_aug, kps_aug = self.seq(image=image, keypoints=kps)
        landmarks_aug = self.kps_to_landmarks(kps_aug)
        landmarks_aug = normalize_landmark(landmarks_aug, img_aug)
        return {'image': img_aug, 'landmarks': landmarks_aug}
    
    def landmarks_to_kps(self, image, landmarks):
        kp_list = []
        for i in range(landmarks.shape[0]):
            kp_list.append(Keypoint(x=landmarks[i][0], y=landmarks[i][1]))
        kps = KeypointsOnImage(kp_list, shape=image.shape)
        return kps

    def kps_to_landmarks(self, kps):
        landmarks = []
        for kp in kps.keypoints:
            landmarks.append((kp.x_int, kp.y_int))
        landmarks = np.array(landmarks)
        return landmarks

class ToTensor(object):
    # def __init__(self, image_size):
    #    self.image_size = image_size

    def __call__(self, sample):
        # w, h = self.image_size
        image, landmarks = sample['image'], sample['landmarks']
        image = image.transpose((2, 0, 1))
        landmarks = landmarks.reshape(-1, 1)
        landmarks = np.squeeze(landmarks).astype('float32')
         
        # normalize image and landmarks to [0, 1]
        return {'image': torch.from_numpy(image).float().div(255),
                 'landmarks': torch.from_numpy(landmarks).float()} 
        
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        return sample

class FaceLandmarkDataset(Dataset):
    def __init__(self, label_dict_list, point_num=106, transform=None):
        self.images = []
        self.landmarks = []
        self.transform = transform
        for label_dict in label_dict_list:
            label_frame = pd.read_csv(label_dict["label_file"], sep=" ", header=None)
            for row in label_frame.iterrows():
                img_path = os.path.join(label_dict['root_dir'], row[1][0])
                landmark = row[1][1:2*point_num+1].values.astype(np.float32).reshape((-1,2))
                # landmark = row[1][1:2*point_num+1].values*config.IMAGE_SIZE
                # landmark = landmark.astype(np.int).reshape((-1,2))
                self.images.append(img_path)
                self.landmarks.append(landmark)
        
        # shuffle
        # landmark_image = list(zip(self.landmarks, self.images))
        # random.shuffle(landmark_image)
        # self.landmarks, self.images = zip(*landmark_image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks = self.landmarks[index]
        sample =  {"image": image, "landmarks": landmarks}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == "__main__":
    test_transform = transforms.Compose([Resize((int(1.4*config.IMAGE_WIDTH), 
                                                 int(1.4*config.IMAGE_HEIGHT))), 
                                       Rescale(3, 15),
                                       # RandomRotate(10),
                                       RandomFlip(0.5),
                                       RandomGaussianBlur(0.6),
                                       RandomMedianBlur(0.6), 
                                       # RandomMotionBlur(20),
                                       RandomCropResize((config.IMAGE_WIDTH, 
                                                         config.IMAGE_HEIGHT), 0.8),
                                       ToTensor(),
                                       Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                                       )
    testset = FaceLandmarkDataset([{'root_dir': config.TRAIN_DATA_DIR,
                                    'label_file': config.LANDMARKS_ANNO_FILE}],
                                    point_num=config.NUM_LANDMARKS, 
                                    transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=1)

    for sample in test_loader:
        image = sample['image'][0]
        print(image.shape)
        landmark = sample['landmarks'][0]
        show_tensor_landmarks(image, landmark)
