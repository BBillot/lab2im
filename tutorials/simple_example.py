"""
Very simple script showing how to generate new images with random contrast


If you use this code, please cite the first SynthSeg paper:
https://github.com/BBillot/lab2im/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


import os
from lab2im import utils
from lab2im.image_generator import ImageGenerator


# path of the input label map
path_label_map = './data_example/brain_label_map.nii.gz'
# path where to save the generated image
result_dir = './generated_images'

# generate an image from the label map.
# Because the image is spatially deformed, we also output the corresponding deformed label map.
brain_generator = ImageGenerator(path_label_map)
im, lab = brain_generator.generate_image()

# save output image and label map
utils.mkdir(result_dir)
utils.save_volume(im, brain_generator.aff, brain_generator.header, os.path.join(result_dir, 'brain.nii.gz'))
utils.save_volume(lab, brain_generator.aff, brain_generator.header, os.path.join(result_dir, 'labels.nii.gz'))
