# This script shows a usecase of the ImageGenerator wrapper around the lab2im_model.
# It generates 5 synthetic brain MRI scans of random contrast from an input label map.
# In this example, the parameters of the GMM (means and variances) are randomly sampled from uniform prior distributions
# of default ranges. We sample the GMM parameters for each new image, so they all have different contrasts.
# Importantly, this example shows how to group labels into classes, so that they are associated with the same Gaussian
# during the image generation.

import os
from lab2im.utils import save_volume
from lab2im.image_generator import ImageGenerator


# label map to generate images from.
# If you have several label maps, BrainGenerator also accepts the path to a directory that contains them.
path_label_map = './data_example/brain_label_map.nii.gz'

# general parameters
n_examples = 5
result_dir = './generated_images'

# By default, ImagGenerator compiles a list of all the labels present in the input label maps, and all these labels are
# used in the generation process.
# This list of labels can also be specified, either to provide the set of labels from which the image should be
# generated, or simply to save some computation time.
# Here we provide it as the path to a 1d numpy array, but it can also be a sequence or directly a 1d numpy array.
generation_labels = './data_example/generation_labels.npy'

# By default, the output label maps contain all the labels used for generation. We can also chose to keep only a subset
# of those, by specifying them in output_labels. This should only contain label already present in the label maps (or in
# generation_labels if it is provided).
output_labels = './data_example/segmentation_labels.npy'

# By default, each label will be associated to a Gaussian distribution when sampling a new image. We can also group
# labels in classes, to force them to share the same Gaussian. This can be done by providing generation classes, which
# should be a sequence, a 1d numpy array, or the path to such an array, with the *same length* as generation_labels.
# Values in generation_classes should be between 0 and K-1, where K is the total number of classes.
generation_classes = './data_example/generation_classes.npy'

########################################################################################################

# instantiate BrainGenerator object
brain_generator = ImageGenerator(labels_dir=path_label_map,
                                 generation_labels=generation_labels,
                                 output_labels=output_labels,
                                 generation_classes=generation_classes)

# create result dir
if not os.path.exists(os.path.join(result_dir)):
    os.mkdir(result_dir)

for n in range(n_examples):

    # generate new image and corresponding labels
    im, lab = brain_generator.generate_image()

    # save output image and label map
    save_volume(im, brain_generator.aff, brain_generator.header, os.path.join(result_dir, 'brain_%s.nii.gz' % n))
    save_volume(lab, brain_generator.aff, brain_generator.header, os.path.join(result_dir, 'labels_%s.nii.gz' % n))
