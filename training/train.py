from ctypes import resize
import os, sys
sys.path.append('.')

import torch 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from data import test

import config

from tools.logger import Logger
from nets.optimized_landmark import LNet
from training.trainer import Trainer
from checkpoint import CheckPoint


use_cuda = config.USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

transform = transforms.Compose([
    transforms.Resize((int(1.2 * config.IMAGE_WIDTH), int(1.2 * config.IMAGE_HEIGHT))),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop((config.IMAGE_HEIGHT, config.IMAGE_WIDTH), scale=(0.8, 1.2), ratio=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


train_dataset = test(
    [{'root_dir': config.TRAIN_DATA_DIR, 'label_file': config.LANDMARKS_ANNO_FILE}],
    point_num=config.NUM_LANDMARKS,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    num_workers=config.num_threads,
    shuffle=True
)
model = LNet()
model.load_state_dict(torch.load('result/iris_lnet/check_point/32_landmarks_model_200.pth'))
if torch.cuda.device_count() > 1:
    print("Train on ", torch.cuda.device_count(), " GPUs")
    nn.DataParallel(model)
model.to(device)

lossfn = nn.MSELoss()
checkpoint = CheckPoint(config.save_path)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.step, gamma=0.1)

logger = Logger(config.save_path)
trainer = Trainer(config.learning_rate, train_loader, model, optimizer, 
    lossfn, scheduler, logger, device, config.save_path)

for epoch in range(1, config.nEpochs+1):
    trainer.train(epoch)
    checkpoint.save_model(model, index=epoch, tag=str(config.NUM_LANDMARKS)+'_landmarks')


