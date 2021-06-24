import cv2
import numpy as np
import os
import math

class DensityAnnotator:
    def __init__(self):
        pass

    def load_image(self, imgr, imgl):
        self.imgr = imgr.copy()
        self.imgl = imgl.copy()
        self.stacked = np.hstack((self.imgl, self.imgr))

    def mouse_callback(self, event, x, y, flags, param):
        cv2.imshow("pixel_selector", self.stacked)
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.right_less_dense = x > 640
            print (x, y)
            cv2.circle(self.stacked, (x, y), 3, (255, 0, 0), -1)

    def run(self, imgr, imgl):
        self.load_image(imgr, imgl)
        self.right_less_dense = None
        cv2.namedWindow('pixel_selector')
        cv2.setMouseCallback('pixel_selector', self.mouse_callback)
        while True: # annotate which one is LESS DENSE
            k = cv2.waitKey(20) & 0xFF
            if k == 27 or self.right_less_dense is not None:
                print(self.right_less_dense)
                break
            if cv2.waitKey(33) == ord('s'):
                print('Skipped current image')
                return None
        return [self.right_less_dense]

if __name__ == '__main__':
    pixel_selector = DensityAnnotator()

    #image_dir = '/Users/priyasundaresan/Downloads/hairtie_overcrossing_resized'
    #image_dir = '/Users/priyasundaresan/Downloads/overhead_hairtie_random_fabric_resized'
    image_dir = '/Users/jennifergrannen/Documents/Berkeley/projects/rope/texas_nonplanar/nonplanar-blue-jpg_train'

    # image_dir = 'single_knots'
    output_dir = 'density_train' # Will have real_data/images and real_data/annots
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    annots_output_dir = os.path.join(output_dir, 'annots')
    images_output_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(annots_output_dir):
        os.mkdir(annots_output_dir)
    if not os.path.exists(images_output_dir):
        os.mkdir(images_output_dir)

    new_idx = 0
    all_images = [a for a in os.listdir(image_dir) if ".jpg" in a]

    while True:
        print("Img %d"%new_idx)
        imgr_filename, imgl_filename = np.random.choice(all_images, size=2, replace=False)
        imgr_path, imgl_path = os.path.join(image_dir, imgr_filename), os.path.join(image_dir, imgl_filename)
        imgr = cv2.resize(cv2.imread(imgr_path), (640,480))
        imgl = cv2.resize(cv2.imread(imgl_path), (640,480))
        imgr_outpath = os.path.join(images_output_dir, '%05d_r.jpg'%new_idx)
        imgl_outpath = os.path.join(images_output_dir, '%05d_l.jpg'%new_idx)
        annots_outpath = os.path.join(annots_output_dir, '%05d.npy'%new_idx)
        annots = pixel_selector.run(imgr, imgl)
        print("---")
        if annots is not None:
            cv2.imwrite(imgr_outpath, imgr)
            cv2.imwrite(imgl_outpath, imgl)
            annots = np.array(annots)
            np.save(annots_outpath, annots)
            new_idx += 1
