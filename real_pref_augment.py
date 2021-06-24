import os
import xml.etree.cElementTree as ET
from xml.dom import minidom
import random
import colorsys
import numpy as np
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

AUGS = [
    iaa.LinearContrast((0.95, 1.05), per_channel=0.25),
    iaa.Add((-10, 10), per_channel=False),
    iaa.GammaContrast((0.85, 1.15)),
    #iaa.GaussianBlur(sigma=(0.5, 0.8)),
    # iaa.GaussianBlur(sigma=(1,2)),
    iaa.MultiplySaturation((0.95, 1.05)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.0125*255)),
    #iaa.flip.Fliplr(0.5),
    ]

KPT_AUGS = [
    iaa.AddToHueAndSaturation((-20, 20)),
    iaa.LinearContrast((0.95, 1.05), per_channel=0.25),
    iaa.Add((-10, 10), per_channel=True),
    iaa.GammaContrast((0.95, 1.05)),
    iaa.GaussianBlur(sigma=(0.0, 0.6)),
    # iaa.ChangeColorTemperature((3000,35000)),
    iaa.MultiplySaturation((0.95, 1.05)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.0125*255)),
    iaa.flip.Flipud(0.5),
    sometimes(iaa.Affine(
               scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
               translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
               rotate=(-20, 20), # rotate by -45 to +45 degrees
               shear=(-10, 10), # shear by -16 to +16 degrees
               order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
               cval=(0, 100), # if mode is constant, use a cval between 0 and 255
               mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
           ))
    ]

seq = iaa.Sequential(KPT_AUGS, random_order=True)
# blur = iaa.Sequential([iaa.GaussianBlur(sigma=(1,2))], random_order=True)

def augment(input, annots, output_dir_img, output_annot_dir, img_filename, show=False):
    width, height, channels = input.shape
    img = input if channels == 3 else input[:, :, :3]
    # img = img.astype(np.uint8)

    # img_blur = blur(image=img)
    img_aug = seq(image=img)
    if show:
        cv2.imshow("img", img_aug)
        cv2.waitKey(0)

    if channels == 3:
        cv2.imwrite(os.path.join(output_dir_img, img_filename), img_aug)
    else:
        combined = img_aug
        for i in range(3, channels):
            gauss = input[:, :, i].reshape((width, height, 1))
            combined = np.append(combined, gauss, axis=2)
        np.save(os.path.join(output_dir_img, img_filename), combined)
        print(combined.shape)
    # np.save(os.path.join(output_annot_dir, img_filename), annots)


if __name__ == '__main__':
    img_dir = 'train/images'
    annots_dir = 'train/annots'
    # img_dir = "datasets/hand_labelled/density/density_train/images"
    # annots_dir = "datasets/hand_labelled/density/density_train/annots"
    if os.path.exists("./aug"):
        os.system("rm -rf ./aug")
    os.makedirs("./aug")
    os.makedirs("./aug/images")
    os.makedirs("./aug/annots")
    output_dir_img = "aug/images"
    output_annot_dir = "aug/annots"
    idx = len(os.listdir(annots_dir))
    orig_len = len(os.listdir(annots_dir))
    num_augs_per = 14
    new_idx = 0
    for i in range(orig_len):
        print(i, orig_len)
        img_r = cv2.imread(os.path.join(img_dir, '%05d_r.jpg'%i))
        img_l = cv2.imread(os.path.join(img_dir, '%05d_l.jpg'%i))
        # img = np.load(os.path.join(img_dir, '%05d.npy'%i), allow_pickle=True)
        annots = np.load(os.path.join(annots_dir, '%05d.npy'%i), allow_pickle=True)
        cv2.imwrite(os.path.join(output_dir_img, "%05d_r.jpg"%new_idx), img_r)
        cv2.imwrite(os.path.join(output_dir_img, "%05d_l.jpg"%new_idx), img_l)
        # np.save(os.path.join(output_dir_img, '%05d.npy'%new_idx), img)
        np.save(os.path.join(output_annot_dir, '%05d.npy'%new_idx), annots)
        new_idx += 1
        for _ in range(num_augs_per):
            img_r_name = "%05d_r.jpg"%new_idx
            img_l_name = "%05d_l.jpg"%new_idx
            augment(img_r, annots, output_dir_img, output_annot_dir, img_r_name, show=False)
            augment(img_l, annots, output_dir_img, output_annot_dir, img_l_name, show=False)
            np.save(os.path.join(output_annot_dir, '%05d.npy'%new_idx), annots)
            new_idx += 1
            idx += 1
        idx -= 1
