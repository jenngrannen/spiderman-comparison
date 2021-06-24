import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, '/host/src')
from torchvision.models import resnet34

class DensityClassifier(nn.Module):
	def __init__(self, img_height=480, img_width=640):
		super(DensityClassifier, self).__init__()
		self.img_height = img_height
		self.img_width = img_width
		self.channels = 6
		self.resnet = resnet34(pretrained=False,
					num_classes=1)
		self.resnet.conv1 = nn.Conv2d(self.channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
		#self.sigmoid = torch.nn.Sigmoid()
	def forward(self, x):
		output = self.resnet(x) 
		return output
                #heatmaps = self.sigmoid(output[:,:self.num_keypoints, :, :])
		#return heatmaps

if __name__ == '__main__':
	model = DensityClassifier().cuda()
	x = torch.rand((1,6,480,640)).cuda()
	result = model.forward(x)
	print(x.shape)
	print(result.shape)
