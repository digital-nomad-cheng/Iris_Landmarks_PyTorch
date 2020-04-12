import os
import time
import random

import cv2
import torch

from tools.utils import AverageMeter, draw_tensor_landmarks


class Trainer(object):
    "Trainer class for only one epoch"    
    def __init__(self, lr, train_loader, model, optimizer, lossfn, scheduler, logger, device, save_path):
        self.lr = lr
        self.train_loader = train_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.lossfn = lossfn.to(self.device)
        self.logger = logger
        self.save_path = save_path
        self.scalar_info = {}
        self.run_count = 0

    def compute_metrics(self, pred_landmarks, gt_landmarks):
       pass

    def train(self, epoch):
        self.scheduler.step()
        self.model.train()
        landmark_loss_ = AverageMeter()

        for batch_idx, sample in enumerate(self.train_loader):
            image = sample['image']
            gt_landmarks = sample['landmarks']
            image, gt_landmarks = image.to(self.device), gt_landmarks.to(self.device)

            pred_landmarks = self.model(image)
            loss = self.lossfn(pred_landmarks, gt_landmarks) 
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            landmark_loss_.update(loss, image.size(0))
            if batch_idx % 20 == 0:
                print("Train Epoch: {:03} [{:05}/{:05} ({:03.0f}%)]\tLoss:{:.6f} LR: {:.7f}".format(
                    epoch, batch_idx*len(sample['image']), len(self.train_loader.dataset), 
                    100.*batch_idx / len(self.train_loader), loss.item(), self.optimizer.param_groups[0]['lr']))
            
        
        self.scalar_info['loss'] = landmark_loss_.avg
        self.scalar_info['lr'] = self.scheduler.get_lr()[0]

        if self.logger is not None:
            for tag, value in list(self.scalar_info.items()):
                self.logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}

        self.run_count += 1

        print("|===>Loss: {:.4f}".format(landmark_loss_.avg))
        
        self.evaluate(epoch, image, gt_landmarks, pred_landmarks)
    
    def evaluate(self, epoch, image, gt_landmarks, pred_landmarks):
        """sample image and landmarks during training"""
        idx = random.randint(0, len(image)-1)
        image = image[idx]
        gt_landmarks = gt_landmarks[idx]
        pred_landmarks = pred_landmarks[idx]
        gt_landmarks = gt_landmarks.cpu().detach()
        pred_landmarks = pred_landmarks.cpu().detach()
        image = image.cpu().detach()
        sample = draw_tensor_landmarks(image, pred_landmarks, gt_landmarks)
        if not os.path.exists(os.path.join(self.save_path, 'sample')):
            os.makedirs(os.path.join(self.save_path, 'sample'))
        cv2.imwrite(os.path.join(self.save_path, 'sample', "epoch_{:03}.jpg".format(epoch)), sample)

