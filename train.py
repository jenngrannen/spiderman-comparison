import pickle
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from config import *
from src.model import DensityClassifier
from src.dataset import DensityDataset, transform
MSE = torch.nn.MSELoss()
bceLoss = nn.BCELoss()

os.environ["CUDA_VISIBLE_DEVICES"]="7"

def forward(sample_batched, model):
    img, gt_label = sample_batched

    img = Variable(img.cuda() if use_cuda else img)
#    img = img.cuda()
    pred_label = model.forward(img).double()
    #pred_gauss = pred_gauss.view(pred_gauss.shape[0], 4, 640*480).double()
    #gt_gauss += 1e-300
    #loss = F.kl_div(gt_gauss.cuda().log(), pred_gauss, None, None, 'mean')
#    loss = nn.BCELoss()(pred_label, gt_label)
    loss = F.binary_cross_entropy_with_logits(pred_label, gt_label)
    return loss

def fit(train_data, test_data, model, epochs, checkpoint_path = ''):
    for epoch in range(epochs):

        train_loss = 0.0
        for i_batch, sample_batched in enumerate(train_data):
            optimizer.zero_grad()
            loss = forward(sample_batched, model)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i_batch + 1, loss.item()), end='')
            print('\r', end='')
        print('train loss:', train_loss / i_batch)
        
        test_loss = 0.0
        for i_batch, sample_batched in enumerate(test_data):
            loss = forward(sample_batched, model)
            test_loss += loss.item()
        print('test loss:', test_loss / i_batch)
        if epoch%2 == 0:
            torch.save(density_model.state_dict(), checkpoint_path + '/model_2_1_' + str(epoch) + '.pth')

# dataset
workers=0
dataset_dir = 'density'
output_dir = 'checkpoints'
save_dir = os.path.join(output_dir, dataset_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

train_dataset = DensityDataset('data/%s/train/images'%dataset_dir,
                           'data/%s/train/annots'%dataset_dir, IMG_HEIGHT, IMG_WIDTH, transform)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

test_dataset = DensityDataset('data/%s/test/images'%dataset_dir,
                           'data/%s/test/annots'%dataset_dir, IMG_HEIGHT, IMG_WIDTH, transform)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

use_cuda = torch.cuda.is_available()
print("USE CUDA", use_cuda)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if use_cuda:
    torch.cuda.set_device(0)

# model
density_model = DensityClassifier(img_height=IMG_HEIGHT, img_width=IMG_WIDTH).cuda()

# optimizer
optimizer = optim.Adam(density_model.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
#optimizer = optim.Adam(density_model.parameters(), lr=0.0001)

fit(train_data, test_data, density_model, epochs=epochs, checkpoint_path=save_dir)
