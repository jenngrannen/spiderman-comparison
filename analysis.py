import pickle
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
from src.model import DensityClassifier
from src.dataset import DensityDataset, transform
from src.prediction import Prediction
from datetime import datetime
from PIL import Image
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="5"

if __name__ == "__main__":
    # model
    density_model = DensityClassifier(img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    density_model.load_state_dict(torch.load('checkpoints/density/model_2_1_24.pth'))

    # cuda
    use_cuda = torch.cuda.is_available()
#    use_cuda = False
    if use_cuda:
        torch.cuda.set_device(0)
        density_model = density_model.cuda()

    prediction = Prediction(density_model, IMG_HEIGHT, IMG_WIDTH, use_cuda)
    transform = transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset_dir = 'hand_labelled/density/density_test'
    test_dataset = DensityDataset('data/%s/images'%dataset_dir,
                           'data/%s/annots'%dataset_dir, IMG_HEIGHT, IMG_WIDTH, transform)
 
    for i, data in enumerate(test_dataset):
        img_t, label = data
        img = img_t.numpy()
        img_t = img_t.cuda()
        # GAUSS
        label = prediction.predict(img_t)
        label = label.detach().cpu().numpy()
        prediction.plot(img, label, image_id=i)
 
