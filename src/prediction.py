import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import numpy as np

class Prediction:
    def __init__(self, model, img_height, img_width, use_cuda):
        self.model = model
        self.img_height  = img_height
        self.img_width   = img_width
        self.use_cuda = use_cuda
        
    def predict(self, imgs):
        # img: torch.Tensor(3, height, width) 
        if len(imgs.shape) == 4:
            imgs = imgs.view(-1, imgs.shape[1], imgs.shape[2], imgs.shape[3])    
        elif len(imgs.shape) == 3:
            imgs = imgs.view(-1, imgs.shape[0], imgs.shape[1], imgs.shape[2])
            
        label = self.model.forward(Variable(imgs))
        return label

    def plot(self, input, label, image_id=0, cls=None, classes=None):
        img_r = input[:3, :, :] * 255
        img_r = img_r.astype(np.uint8)
        img_l = input[3:, :, :] * 255
        img_l = img_l.astype(np.uint8)
        img_r, img_l = np.transpose(img_r, (1,2,0)).copy(), np.transpose(img_l, (1,2,0)).copy() 
        print("Running inferences on image: %d"%image_id)
        if label < 0.5:
            cv2.putText(img_l, "Less Dense", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        else:
            cv2.putText(img_r, "Less Dense", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        result = cv2.hconcat((img_l, img_r))
        cv2.imwrite('preds/out%04d.png'%image_id, result)
