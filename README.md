# SPiDERMan Image Comparison

* This repo provides an implementation of pairwise image comparison using a ResNet backbone in PyTorch. We use this model to sensing untangling progress and plan recovery actions with SPiDERMan: Sensing Progress in Dense Entanglements for Recovery Manipulation. For more details, see:

[**"Untangling Dense Non-Planar Knots by Learning Manipulation Features and Recovery Policies"**](https://sites.google.com/berkeley.edu/non-planar-untangling)

__Priya Sundaresan*, Jennifer Grannen*, Brijen Thananjeyan, Ashwin Balakrishna, Jeffrey Ichnowski, Ellen Novoseller, Minho Hwang, Michael Laskey, Joseph E. Gonzalez, Ken Goldberg (*equal contribution)__

### Description
  * `docker`: Contains utils for building Docker images and Docker containers to manage Python dependencies used in this project 
  * `src`: Contains model definitions, dataloaders, and visualization utils
  * `config.py`: Script for configuring hyperparameters for a training job
  * `train.py`: Script for training
  * `analysis.py`: Script for running inference on a trained model and saving predicted preference visualizations
  * `annotate_real_density.py`: Script for hand-labelling and saving paired image preferences

### Getting Started/Overview
#### Dataset Generation
* We provide a sample dataset, `density_dataset` which contains a train and test dataset of images of knots in single cable settings and their corresponding annotations comparing the density across two configurations (labelled True if if `{img_num}_r.jpg` is less dense than `{img_num}_l.jpg`). 
* When hand-labelling a datset, use the script `python annotate_real_density.py` which expects the path to a folder of jpg images in `image_dir`. It will launch an OpenCV window where you can annotate preferences; double click on the image that is "preferred" (less dense).  Note that the script will automatically go to the next image once 1 click is recorded. Press `s` to skip an image. The script saves the images/annotations to a folder specifcied in `output_dir` organized as follows:
```
{output_dir}/
|-- images
|   `-- 00000_l.jpg
|   `-- 00000_r.jpg
|   ...
`-- annots
    `-- 00000.npy
    ...
```
* Use the script `real_pref_augment.py` which expects two folders with `images` and `annots` (copy `{output_dir}/images` and `{output_dir}/keypoints` to the same directory level as this script). It will use image space and affine transformations to augment the dataset by `num_augs_per` and output the augmented images and copied annotations to `aug/images` and `aug/annots` folders
* Repeat the above steps for a train and test set. This should produce a dataset like so:
```
<your_dataset_name>
|-- test
|   |-- images
|   `-- annots
`-- train
    |-- images
    `-- annots
```

#### Training and Inference
* Start a docker container (if not already done) with `cd docker && ./docker_run.py`
* Configure `train.py` by replacing `dataset_dir = 'density'` with `<your_dataset_name>`
* Run `python train.py`
* This will save checkpoints to `checkpoints/<your_dataset_name>`
* Update `analysis.py` by with `keypoints.load_state_dict(torch.load('checkpoints/<your_dataset_name>/model_2_1_24.pth'))`
* Run `python analysis.py` which will save predicted preference labels to `preds`

### Contributing 
* For any questions, contact [Priya Sundaresan](http://priya.sundaresan.us) at priya.sundaresan@berkeley.edu or [Jennifer Grannen](http://jenngrannen.com/) at jenngrannen@berkeley.edu
