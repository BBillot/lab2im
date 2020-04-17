# python imports
import os
import time
import logging
import numpy as np

# project imports
import utils
from labels_to_image_model import labels_to_image_model
from model_input_generator import build_model_input_generator


class BrainGenerator:

    def __init__(self,
                 labels_dir,
                 generation_label_list,
                 segmentation_label_list,
                 generation_n_neutral_labels,
                 n_channels=1,
                 classes_list=None,
                 target_res=None,
                 blurring_res=None,
                 padding_margin=None,
                 cropping=None,
                 means_range=None,
                 std_devs_range=None,
                 one_stats_for_channel=True,
                 intensity_constraints='rl_grouping',
                 apply_affine_trans=True,
                 scaling_range=None,
                 rotation_range=None,
                 shearing_range=None,
                 apply_nonlin_trans=True,
                 nonlin_shape_factor=0.0625,
                 nonlin_std_dev=3.,
                 blur_range=1.15,
                 blur_background=True,
                 thickness=None,
                 downsample=False,
                 apply_bias_field=True,
                 bias_shape_factor=0.025,
                 bias_field_std_dev=0.3,
                 crop_channel_2=None,
                 backgrounds_dir=None,
                 apply_normalisation=True,
                 output_div_by_n=None,
                 flipping=True,
                 head_model=False,
                 batch_size=1,
                 in_memory=False):
        """
        This class is wrapper around the brain generative model. It contains both the GPU model that performs generation
         and augmentation, as well as a python generator that returns the input data for the generative model.
        To generate pairs of image/labels you can just call the method generate_brain() on an object of this class. It
        will call the internally defined model input generator and feed it to the generative model.
        You can also get the generative model by using the argument 'labels_to_image_model', and the model input
        generator with argument 'model_inputs_generator'
        :param labels_dir (str): path of folder containing the poll of labels to chose from
        :param generation_label_list (list or 1 dimensional array): list of all the labels contained in the label maps
        to generate the synthetic images from. The list have to be ordered as such: first all neutral labels including
        background, then all left labels and finally corresponding right labels.
        ex: [0 (bground), 24 (CSF), 2 (left WM), 3 (left GM), 41 (right WM), 42 (right GM)].
        :param segmentation_label_list (list or 1 dimensional array): list of all the labels to keep in the output label
        maps. This list has to be ordered in the same way as generation_label_list, and to only contain labels already
        contained in generation_label_list.
        :param n_generation_neutral_labels (int): number of neutral labels in label_list.
        :param n_channels (int): channel number of synthetised images.
        :param classes_list (list or 1 dimensional array): intensity classes of each label, same length and order as
        label_list. Labels with same class will share the same distribution. Default is None, where only right/left
        labels are automatically grouped together.
        :param target_res (float or list): target resolution (in mm). If float all dimension will be resampled at
        the same resolution. If None, the outputs will have the same resolution as input.
        :param padding_margin (float or list): margin by which the input labels will be 0-padded. This step happens
        before an eventual cropping. Default is None, no padding.
        :param cropping (float or list): impose the shape of the generator's output by cropping.
        :param means_range: range from which the mean intensity of each label will be drawn. Can be list of 2
        [min max] that will be shared by all labels, or an array of shape (2, n_labels). In the case of an array,
        the mean of each structure will be drawn from normal distribution with means defined in the first row and std
        dev in the second. means_range also support arrays of 2*n_mod rows where stats of different modalities are
        concatenated in the first dimension. In that case the modality is randomly drawn for each image.
        Default is None, which corresponds to means_range = [25, 225].
        :param std_devs_range: range from which the mean intensity of each label will be drawn. Can be list of 2
        [min max] that will be shared by all labels, or an array of shape (2, n_labels). In the case of an array,
        the std dev of each structure will be drawn from normal distribution with means defined in the first row and std
        dev in the second. std_devs_range also support arrays of 2*n_mod rows where stats of different modalities are
        concatenated in the first dimension. In that case the modality is randomly drawn for each image.
        Default is None, which corresponds to means_range = [25, 225].
        :param one_stats_for_channel: if n_channel = n_mod (cf means_range), retrieve specific stats for each channel.
        Channels will use the stats in the order they are provided in means_range and std_devs_range. Default is True.
        :param intensity_constraints (str): type of constraints to apply to means/variances for image generation:
        'no_rules', 'rl_grouping': corresponding right left structures are grouped together, 'classes': labels with
        same class share the same stats (necessitate classes_list), 'classes_with_relations': intensity are defined with
        WM GM ans CSF, 'stats': need to provide means/variances for each label.
        :param apply_affine_deformation (bool): whether to apply affine deformation. (default = True)
        :param scaling_range (list): if apply_affine_deformation is True, range from which scaling parameters will be
        drawn. If int or float, the range is [-scaling_range, scaling_range].
        If None (default), scaling_range = [0.93, 1.07]
        :param rotation_range: if apply_affine_deformation is True, range from which rotation angles (degree)
        will be drawn. If int or float, the rotation will be the same for all dimensions, taken from uniform
        distribution symmetric with respect to 0. If list of 2, it will be drawn uniformly between the two values.
        If numpy array (size 2*n_dims), first row specifies lower bound for each dim, and second row specifies upper
        bound. If None (default), rotation_range = [-10, 10].
        :param shearing_range: if apply_affine_deformation is True, range from which shearing parameters will be
        drawn. If int or float, the range is [-shearing_range, shearing_range].
        If None (default), shearing_range = [-0.01, 0.01]
        :param apply_nonlin_deformation (bool): whether to apply non linear deformation. (default = True).
        Obtained from a first field sampled from normal distribution, resized and integrated.
        :param nonlin_shape_factor (float): If apply_nonlin_deformation=True, ratio between labels shape and the shape
        of the first sampled field. (default 0.0625)
        :param nonlin_std_dev (float): If apply_nonlin_deformation=True, standard deviation of the normal distribution
        from which we sample the first field (default 3)
        :param blur_range: Randomise blurring_res. Each element of blurring_res is multiplied at each mini_batch by a
        random coef sampled from a uniform distribution with bounds [1-blur_range, 1+blur_range].
        Default is 0.15. If blur_range is None, no randomisation.
        :param blur_background (bool): Whether background is a regular label, thus blurred with the others.
        If not the gaussian blurring is masked, and background follows a normal distribution with mean and std dev
        between 0 and 10.
        :param thickness: Size (in mm) of slices (int) or in each dimension (list). (default is None)
        :param apply_bias_field: whether to apply a bias field to the final image (default True).
        Obtained from a first field sampled from normal distribution.
        :param bias_shape_factor (float): If apply_nonlin_deformation=True, ratio between labels shape and the shape
        of the first sampled field. (default 0.025)
        :param bias_field_std_dev (float): If apply_bias_field=True, standard deviation of the normal distribution
        from which we sample the first field (default 0.3).
        :param crop_channel_2: stats for cropping second channel along the anterior-posterior axis.
        Should be a vector of length 4, with bounds of uniform distribution for cropping the front and back of the image
        (in percentage). None is no croppping.
        :param backgrounds_dir (str): path of folder containing background labels on which we paste randomly selected
        training labels. These background labels must be of the same size as the training labels. Default is None, where
        background labels are not used.
        :param apply_normalisation: whether to normalise data
        :param output_div_by_n: if not None, make the shape of the output image divisible by this value
        :param flipping: whether to randomly apply flipping (right/left only) to labels. (default True)
        :param head_model: whether training labels contain head labels (extracerebral, optic chiasm, etc) (default Fals)
        :param batch_size: numbers of brains generated at the same time (default 1).
        :param in_memory:  whether to load the full labels dataset in memory (default False)
        """

        # prepare data files
        if ('.nii.gz' in labels_dir) | ('.mgz' in labels_dir) | ('.npz' in labels_dir):
            self.labels_paths = [labels_dir]
        else:
            self.labels_paths = utils.list_images_in_folder(labels_dir)
        assert len(self.labels_paths) > 0, "Could not find any training data"
        # prepare background files
        if backgrounds_dir is not None:
            if ('.nii.gz' in backgrounds_dir) | ('.mgz' in backgrounds_dir) | ('.npz' in backgrounds_dir):
                self.background_paths = [backgrounds_dir]
            else:
                self.background_paths = utils.list_images_in_folder(backgrounds_dir)
            assert len(self.background_paths) > 0, "Could not find any training data"
        else:
            self.background_paths = None

        # read info from image
        self.labels_shape, self.aff, self.n_dims, _, self.header, self.atlas_res = \
            utils.get_volume_info(self.labels_paths[0])
        self.n_channels = n_channels
        self.generation_label_list = generation_label_list
        self.segmentation_label_list = segmentation_label_list
        self.generation_n_neutral_labels = generation_n_neutral_labels
        self.classes_list = utils.load_array_if_path(classes_list)
        self.target_res = utils.load_array_if_path(target_res)
        self.data_res = utils.load_array_if_path(blurring_res)

        # augmentation parameters
        self.means_range = utils.load_array_if_path(means_range)
        self.std_devs_range = utils.load_array_if_path(std_devs_range)
        self.intensity_constraints = intensity_constraints
        self.padding_margin = utils.load_array_if_path(padding_margin)
        self.crop = utils.load_array_if_path(cropping)
        self.downsample = downsample
        self.apply_affine_trans = apply_affine_trans
        self.scaling_range = utils.load_array_if_path(scaling_range)
        self.rotation_range = utils.load_array_if_path(rotation_range)
        self.shearing_range = utils.load_array_if_path(shearing_range)
        self.apply_nonlin_trans = apply_nonlin_trans
        self.nonlin_shape_factor = nonlin_shape_factor
        self.nonlin_std_dev = nonlin_std_dev
        self.blur_range = blur_range
        self.blur_background = blur_background
        self.thickness = utils.load_array_if_path(thickness)
        self.apply_bias_field = apply_bias_field
        self.bias_shape_factor = bias_shape_factor
        self.bias_field_std_dev = bias_field_std_dev
        self.crop_second_channel = utils.load_array_if_path(crop_channel_2)
        self.normalise = apply_normalisation
        self.output_div_by_n = output_div_by_n
        self.flipping = flipping

        # build transformation model
        self.labels_to_image_model, self.model_output_shape = self._build_labels_to_image_model()

        # build generator for model inputs
        self.model_inputs_generator = self._build_model_inputs_generator(one_stats_for_channel,
                                                                         batch_size,
                                                                         in_memory,
                                                                         head_model)

        # build brain generator
        self.brain_generator = self._build_brain_generator()

    def _build_labels_to_image_model(self):
        # build_model
        lab_to_im_model, out_shape = labels_to_image_model(self.labels_shape,
                                                           self.n_channels,
                                                           self.atlas_res,
                                                           self.target_res,
                                                           self.data_res,
                                                           self.crop,
                                                           self.generation_label_list,
                                                           self.segmentation_label_list,
                                                           self.generation_n_neutral_labels,
                                                           self.aff,
                                                           padding_margin=self.padding_margin,
                                                           apply_affine_trans=self.apply_affine_trans,
                                                           apply_nonlin_trans=self.apply_nonlin_trans,
                                                           nonlin_shape_factor=self.nonlin_shape_factor,
                                                           blur_range=self.blur_range,
                                                           blur_background=self.blur_background,
                                                           thickness=self.thickness,
                                                           downsample=self.downsample,
                                                           apply_bias_field=self.apply_bias_field,
                                                           bias_shape_factor=self.bias_shape_factor,
                                                           crop_channel2=self.crop_second_channel,
                                                           normalise=self.normalise,
                                                           output_div_by_n=self.output_div_by_n,
                                                           flipping=self.flipping)
        return lab_to_im_model, out_shape

    def _build_model_inputs_generator(self, one_stats_for_channel, batch_size, in_memory, head_model):
        # build model's inputs generator
        model_inputs_generator = build_model_input_generator(self.labels_paths,
                                                             self.generation_label_list,
                                                             self.generation_n_neutral_labels,
                                                             self.labels_shape,
                                                             self.model_output_shape,
                                                             self.n_channels,
                                                             classes_list=self.classes_list,
                                                             means_range=self.means_range,
                                                             std_devs_range=self.std_devs_range,
                                                             use_specific_stats_for_channel=one_stats_for_channel,
                                                             intensity_constraints=self.intensity_constraints,
                                                             apply_affine_trans=self.apply_affine_trans,
                                                             scaling_range=self.scaling_range,
                                                             rotation_range=self.rotation_range,
                                                             shearing_range=self.shearing_range,
                                                             apply_nonlin_trans=self.apply_nonlin_trans,
                                                             nonlin_shape_fact=self.nonlin_shape_factor,
                                                             nonlin_std_dev=self.nonlin_std_dev,
                                                             apply_bias_field=self.apply_bias_field,
                                                             bias_shape_fact=self.bias_shape_factor,
                                                             bias_field_std_dev=self.bias_field_std_dev,
                                                             blur_background=self.blur_background,
                                                             background_paths=self.background_paths,
                                                             head=head_model,
                                                             batch_size=batch_size,
                                                             in_memory=in_memory)
        return model_inputs_generator

    def _build_brain_generator(self):
        while True:
            model_inputs = next(self.model_inputs_generator)
            [image, labels] = self.labels_to_image_model.predict(model_inputs)
            yield image, labels

    def generate_brain(self):
        """call this method when an object of this class has been instantiated to generate new brains"""
        (image, labels) = next(self.brain_generator)
        return image, labels


if __name__ == '__main__':
    logging.getLogger('tensorflow').disabled = True

    # -------------------------------------------------- full brain ----------------------------------------------------

    # # path training labels directory (can also be path of a single image) and result folder
    # paths = '../hippocampus_seg/atlases_full_brain'
    # result_folder = '../PVSeg/generated_images/b40_test'
    #
    # # general parameters
    # n_examples = 10
    # batchsize = 1
    # channels = 1
    # specific_stats_for_channel = True
    # target_resolution = None  # in mm
    # padding = None  # pad labels at the beginning of generation, done before cropping
    # crop = 160  # crop produced image to this size
    # flip = True
    # output_divisible_by_n = 32  # output image should have dimension divisible by n (e.g. for deep learning use)
    # constraints_intensity = 'classes_with_stats'  # type of constraint
    # head = False
    # background_blur = True
    # nonlin_std = 3
    # blurring_range = 1.5
    # bias_field_std = 0.6
    # range_scaling = None
    # range_shearing = None
    # downsample_data = True
    #
    # # list of all labels in training label maps, can also be computed and saved
    # load_generation_label_list = '../PVSeg/labels_classes_stats/mit_generation_labels.npy'
    # load_segmentation_label_list = '../PVSeg/labels_classes_stats/mit_segmentation_labels.npy'
    # save_label_list = None  # set path of computed label list here
    #
    # # optional parameters
    # path_classes_list = '../PVSeg/labels_classes_stats/mit_generation_classes.npy'
    # path_means_range = '../PVSeg/labels_classes_stats/mit_means_range.npy'
    # path_std_devs_range = '../PVSeg/labels_classes_stats/mit_std_devs_range.npy'
    # path_crop_channel_2_range = None
    # path_rotation_range = None
    # path_blurring_resolution = '../PVSeg/labels_classes_stats/mit_blurring_resolution_6_1_1.npy'
    # path_thickness = '../PVSeg/labels_classes_stats/mit_thickness_4_1_1.npy'

    # ----------------------------------------------------- hippo ------------------------------------------------------

    # path training labels directory (can also be path of a single image) and result folder
    paths = '../PVSeg/atlases_hippo'
    result_folder = '../PVSeg/generated_images/test'

    # general parameters
    n_examples = 10
    batchsize = 1
    channels = 2
    specific_stats_for_channel = True
    target_resolution = [0.6, 0.6, 0.6]  # in mm
    padding = None  # pad labels at the beginning of generation, done before cropping
    crop = 96  # crop produced image to this size
    flip = False
    output_divisible_by_n = None  # 16  # output image should have dimension divisible by n (e.g. for deep learning use)
    constraints_intensity = 'classes_with_stats'  # type of constraint
    head = False
    background_blur = False
    nonlin_std = 4
    blurring_range = 1.2
    bias_field_std = 0.5
    range_scaling = '../PVSeg/labels_classes_stats/cobralab_scaling_range.npy'
    range_shearing = 0.015
    downsample_data = True

    # list of all labels in training label maps, can also be computed and saved
    load_generation_label_list = '../PVSeg/labels_classes_stats/cobralab_generation_labels.npy'
    path_segmentation_label_list = '../PVSeg/labels_classes_stats/cobralab_segmentation_labels.npy'
    save_label_list = None  # set path of computed label list here

    # optional parameters
    path_classes_list = '../PVSeg/labels_classes_stats/cobralab_generation_classes.npy'
    path_means_range = '../PVSeg/labels_classes_stats/cobralab_means_range.npy'
    path_std_devs_range = '../PVSeg/labels_classes_stats/cobralab_std_devs_range.npy'
    path_crop_channel_2_range = '../PVSeg/labels_classes_stats/cobralab_cropping_stats_t2.npy'
    path_rotation_range = '../PVSeg/labels_classes_stats/cobralab_rotation_range.npy'
    path_blurring_resolution = '../PVSeg/labels_classes_stats/cobralab_blurring_resolution.npy'
    path_thickness = None

    ########################################################################################################

    # load label list, classes list and intensity ranges if necessary
    generation_list_labels, generation_neutral_labels = utils.get_list_labels(load_generation_label_list, FS_sort=True)
    if path_segmentation_label_list is not None:
        segmentation_list_labels, _ = utils.get_list_labels(path_segmentation_label_list, FS_sort=True)
    else:
        segmentation_list_labels = generation_list_labels

    # instantiate BrainGenerator object
    brain_generator = BrainGenerator(labels_dir=paths,
                                     generation_label_list=generation_list_labels,
                                     segmentation_label_list=segmentation_list_labels,
                                     classes_list=path_classes_list,
                                     means_range=path_means_range,
                                     scaling_range=range_scaling,
                                     shearing_range=range_shearing,
                                     rotation_range=path_rotation_range,
                                     std_devs_range=path_std_devs_range,
                                     one_stats_for_channel=specific_stats_for_channel,
                                     intensity_constraints=constraints_intensity,
                                     generation_n_neutral_labels=generation_neutral_labels,
                                     n_channels=channels,
                                     target_res=target_resolution,
                                     nonlin_std_dev=nonlin_std,
                                     blurring_res=path_blurring_resolution,
                                     blur_range=blurring_range,
                                     padding_margin=padding,
                                     batch_size=batchsize,
                                     cropping=crop,
                                     blur_background=background_blur,
                                     thickness=path_thickness,
                                     downsample=downsample_data,
                                     bias_field_std_dev=bias_field_std,
                                     crop_channel_2=path_crop_channel_2_range,
                                     output_div_by_n=output_divisible_by_n,
                                     apply_normalisation=True,
                                     flipping=flip)

    if not os.path.exists(os.path.join(result_folder)):
        os.mkdir(result_folder)

    for n in range(n_examples):

        # generate new image and corresponding labels
        start = time.time()
        im, lab = brain_generator.generate_brain()
        end = time.time()
        print('deformation {0:d} took {1:.01f}s'.format(n, end - start))

        # save image
        for b in range(batchsize):
            utils.save_volume(np.squeeze(im[b, ...]), brain_generator.aff, brain_generator.header,
                              os.path.join(result_folder, 'minibatch_{}_image_{}.nii.gz'.format(n, b)))
            utils.save_volume(np.squeeze(lab[b, ...]), brain_generator.aff, brain_generator.header,
                              os.path.join(result_folder, 'minibatch_{}_labels_{}.nii.gz'.format(n, b)))
