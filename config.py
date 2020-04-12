
USE_CUDA = True

NUM_LANDMARKS = 32
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 80

# generate data for training

# data for training 
TRAIN_DATA_DIR = './data/train'
LANDMARKS_ANNO_FILE = 'annotations/iris_landmark.txt'

# training config
save_path = './result/iris_lnet'
nEpochs = 200
batch_size = 256
num_threads = 8
learning_rate = 2e-3
step = [20, 80, 160, 180]


