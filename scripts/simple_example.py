# Very simple script showing how to generate new images with lab2im

import os
from lab2im import utils
from lab2im.image_generator import ImageGenerator

path_label_map = '../data_example/brain_label_map.nii.gz'
path_resulr_dir = '../data_example/generated_images'

# generate an image and corresponding labels from a label map
brain_generator = ImageGenerator(path_label_map)
im, lab = brain_generator.generate_image()

# save output image and label map
if not os.path.exists(os.path.join(path_resulr_dir)):
    os.mkdir(path_resulr_dir)
utils.save_volume(im, brain_generator.aff, brain_generator.header, os.path.join(path_resulr_dir, 'image.nii.gz'))
utils.save_volume(lab, brain_generator.aff, brain_generator.header, os.path.join(path_resulr_dir, 'labels.nii.gz'))
