import cv2
import torch
from math import log2
import os

ROOT_CHECKPOINT = 'checkpoint_cleaned-data-secondtry'
START_TRAIN_AT_IMG_SIZE = 4
CHECKPOINT_PATH="checkpoint"
# DATASET = '/mnt/c/Users/Worapob/Desktop/ML_playground/AniFace_GAN/data_preprocess/cleaned_dataset/cleanded_dataset7/'
DATASET = '../IM_Train/cleaned'
CHECKPOINT_GEN = os.path.join(ROOT_CHECKPOINT,"/generator_lastest/generator_lastest.pth")
CHECKPOINT_CRITIC = os.path.join(ROOT_CHECKPOINT,"critic_lastest/critic_lastest.pth")
# CHECKPOINT_GEN = f"checkpoint/generator_lastest/generator_lastest.pth"
# CHECKPOINT_CRITIC = f"checkpoint/critic_lastest/critic_lastest.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BATCH_SIZES = [128, 64, 32, 16, 16, 16, 16, 8, 4]
CHANNELS_IMG = 3
Z_DIM = 512  # should be 512 in original paper
IN_CHANNELS = 512  # should be 512 in original paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [100] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 24
NUM_CLASSES = 10