import os, sys
sys.path.append('.')

import torch 
import torch.nn as nn
from torchvision import transforms

import config
from tools.iris_dataset import * # transformers
from tools.logger import Logger
from nets.optimized_landmark import LNet
from training.trainer import Trainer
from checkpoint import CheckPoint


use_cuda = config.USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

transform = transforms.Compose([Resize((int(1.4*config.IMAGE_WIDTH), 
                                        int(1.4*config.IMAGE_HEIGHT))),
                                Rescale(3, 15),
                                RandomFlip(0.4),
                                RandomGaussianBlur(0.6), 
                                RandomMedianBlur(0.6), 
                                # RandomMotionBlur(20),
                                RandomCropResize((config.IMAGE_WIDTH, 
                                                  config.IMAGE_HEIGHT), 0.2),
                                ToTensor(),
                                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                                )

train_loader = torch.utils.data.DataLoader(
    FaceLandmarkDataset([{'root_dir': config.TRAIN_DATA_DIR,
                         'label_file': config.LANDMARKS_ANNO_FILE}],
                         point_num=config.NUM_LANDMARKS,
                         transform=transform),
    batch_size = config.batch_size,
    num_workers = config.num_threads,
    shuffle=True)

model = LNet()
model.load_state_dict(torch.load('result/iris_lnet/check_point/32_landmarks_model_200.pth'))
if torch.cuda.device_count() > 1:
    print("Train on ", torch.cuda.device_count(), " GPUs")
    nn.DataParallel(model)
model.to(device)

lossfn = nn.MSELoss()
checkpoint = CheckPoint(config.save_path)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.step, gamma=0.1)

logger = Logger(config.save_path)
trainer = Trainer(config.learning_rate, train_loader, model, optimizer, 
    lossfn, scheduler, logger, device, config.save_path)

for epoch in range(1, config.nEpochs+1):
    trainer.train(epoch)
    checkpoint.save_model(model, index=epoch, tag=str(config.NUM_LANDMARKS)+'_landmarks')


