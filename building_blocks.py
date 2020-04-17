# This file contains functions for data augmentation in keras models with a tensorflow backend.
# The functions are regrouped in the following order:
# 1- shape augmentation
# 2- label_to_image
# 3- blurring
# 4- resampling
# 5- intensity augmentation
# 6- edit label maps

# python imports
import math
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
import tensorflow_probability as tfp

# project imports
import utils
from volume_editting import edit_volume

# third-party imports
import ext.neuron.layers as nrn_layers


# -------------------------------------------- SHAPE AUGMENTATION FUNCTIONS --------------------------------------------

def deform_tensor(tensor, affine_trans=None, elastic_trans=None, n_dims=3):
    """This function spatially deforms a tensor with a combination of affine and elastic transformations.
    :param tensor: input tensor to deform
    :param affine_trans: (optional) tensor of shape [?, n_dims+1, n_dims+1] corresponding to an affine transformation.
    Default is None, no affine transformation is applied. Should not be None if elastic_trans is None.
    :param elastic_trans: (optional) tensor of shape [?, x, y, z, n_dims] corresponding to a small-size SVF, that is:
    1) resized to half the shape of volume
    2) integrated
    3) resized to full image size
    Default is None, no elastic transformation is applied. Should not be None if affine_trans is None.
    :param n_dims: (optional) number of dimensions of the initial image (excluding batch and channel dimensions)
    :return: tensor of the same shape as volume
    """

    assert (affine_trans is not None) | (elastic_trans is not None), 'affine_trans or elastic_trans should be provided'

    # reformat image
    tensor._keras_shape = tuple(tensor.get_shape().as_list())
    image_shape = tensor.get_shape().as_list()[1:n_dims + 1]
    tensor = KL.Lambda(lambda x: tf.cast(x, dtype='float'))(tensor)
    trans_inputs = [tensor]

    # add affine deformation to inputs list
    if affine_trans is not None:
        trans_inputs.append(affine_trans)

    # prepare non-linear deformation field and add it to inputs list
    if elastic_trans is not None:
        elastic_trans_shape = elastic_trans.get_shape().as_list()[1:n_dims+1]
        resize_shape = [max(int(image_shape[i]/2), elastic_trans_shape[i]) for i in range(n_dims)]
        nonlin_field = nrn_layers.Resize(size=resize_shape, interp_method='linear')(elastic_trans)
        nonlin_field = nrn_layers.VecInt()(nonlin_field)
        nonlin_field = nrn_layers.Resize(size=image_shape, interp_method='linear')(nonlin_field)
        trans_inputs.append(nonlin_field)

    # apply deformations
    return nrn_layers.SpatialTransformer(interp_method='nearest')(trans_inputs)


def random_cropping(tensor, crop_shape, n_dims=3):
    """Randomly crop an input tensor to a tensor of a given shape. This cropping is applied to all channels.
    :param tensor: input tensor to crop
    :param crop_shape: shape of the cropped tensor, excluding batch and channel dimension.
    :param n_dims: (optional) number of dimensions of the initial image (excluding batch and channel dimensions)
    :return: cropped tensor
    example: if tensor has shape [2, 160, 160, 160, 3], and crop_shape=[96, 128, 96], then this function returns a
    tensor of shape [2, 96, 128, 96, 3], with randomly selected cropping indices.
    """

    # get maximum cropping indices in each dimension
    image_shape = tensor.get_shape().as_list()[1:n_dims + 1]
    cropping_max_val = [image_shape[i] - crop_shape[i] for i in range(n_dims)]

    # prepare cropping indices and tensor's new shape (don't crop batch and channel dimensions)
    crop_idx = KL.Lambda(lambda x: tf.zeros([1], dtype='int32'))([])
    for val_idx, val in enumerate(cropping_max_val):  # draw cropping indices for image dimensions
        if val > 0:
            crop_idx = KL.Lambda(lambda x: tf.concat([tf.cast(x, dtype='int32'), K.random_uniform([1], minval=0,
                                 maxval=val, dtype='int32')], axis=0))(crop_idx)
        else:
            crop_idx = KL.Lambda(lambda x: tf.concat([tf.cast(x, dtype='int32'),
                                                      tf.zeros([1], dtype='int32')], axis=0))(crop_idx)
    crop_idx = KL.Lambda(lambda x: tf.concat([tf.cast(x, dtype='int32'),
                                              tf.zeros([1], dtype='int32')], axis=0))(crop_idx)
    patch_shape_tens = KL.Lambda(lambda x: tf.convert_to_tensor([-1] + crop_shape + [-1], dtype='int32'))([])

    # perform cropping
    tensor = KL.Lambda(lambda x: tf.slice(x[0], begin=tf.cast(x[1], dtype='int32'),
                                          size=tf.cast(x[2], dtype='int32')))([tensor, crop_idx, patch_shape_tens])

    return tensor, crop_idx


def label_map_random_flipping(labels, label_list, n_neutral_labels, vox2ras, n_dims=3):
    """This function flips a label map with a probability of 0.5.
    Right/left label values are also swapped if the label map is flipped in order to preserve the right/left sides.
    :param labels: input label map
    :param label_list: list of all labels contained in labels. Must be ordered as follows, first the neutral labels
    (i.e. non-sided), then left labels and right labels.
    :param n_neutral_labels: number of non-sided labels
    :param vox2ras: affine matrix of the initial input label map, to find the right/left axis.
    :param n_dims: (optional) number of dimensions of the initial image (excluding batch and channel dimensions)
    :return: tensor of the same shape as label map, potentially right/left flipped with correction for sided labels.
    """

    # boolean tensor to decide whether to flip
    rand_flip = KL.Lambda(lambda x: K.greater(tf.random.uniform((1, 1), 0, 1), 0.5))([])

    # swap right and left labels if we later right-left flip the image
    n_labels = len(label_list)
    if n_neutral_labels != n_labels:
        rl_split = np.split(label_list, [n_neutral_labels, int((n_labels - n_neutral_labels) / 2 + n_neutral_labels)])
        flipped_label_list = np.concatenate((rl_split[0], rl_split[2], rl_split[1]))
        labels = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                              KL.Lambda(lambda x: tf.gather(
                                                  tf.convert_to_tensor(flipped_label_list, dtype='int32'),
                                                  tf.cast(x, dtype='int32')))(y[1]),
                                              tf.cast(y[1], dtype='int32')))([rand_flip, labels])
    # find right left axis
    ras_axes, _ = edit_volume.get_ras_axes_and_signs(vox2ras, n_dims)
    flip_axis = [ras_axes[0] + 1]

    # right/left flip
    labels = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                          KL.Lambda(lambda x: K.reverse(x, axes=flip_axis))(y[1]),
                                          y[1]))([rand_flip, labels])

    return labels, rand_flip


def restrict_tensor(tensor, axes, boundaries):
    """Reset the edges of a tensor to zero. This is performed only along the given axes.
    The width of the zero-band is randomly drawn from a uniform distribution given in boundaries.
    :param tensor: input tensor
    :param axes: axes along which to reset edges to zero. Can be an int (single axis), or a sequence.
    :param boundaries: numpy array of shape (len(axes), 4). Each row contains the two bounds of the uniform
    distributions from which we draw the width of the zero-bands on each side.
    Those bounds must be expressed in relative side (i.e. between 0 and 1).
    :return: a tensor of the same shape as the input, with bands of zeros along the pecified axes.
    example:
    tensor=tf.constant([[[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]]])  # shape = [1,10,10,1]
    axes=1
    boundaries = np.array([[0.2, 0.45, 0.85, 0.9]])

    In this case, we reset the edges along the 2nd dimension (i.e. the 1st dimension after the batch dimension),
    the 1st zero-band will expand from the 1st row to a number drawn from [0.2*tensor.shape[1], 0.45*tensor.shape[1]],
    and the 2nd zero-band will expand from a row drawn from [0.85*tensor.shape[1], 0.9*tensor.shape[1]], to the end of
    the tensor. A possible output could be:
    array([[[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]])  # shape = [1,10,10,1]
    """

    shape = tuple(tensor.get_shape().as_list())
    axes = utils.reformat_to_list(axes, dtype='int')
    boundaries = utils.reformat_to_n_channels_array(boundaries, n_dims=4, n_channels=len(axes))

    # build mask
    mask = KL.Lambda(lambda x: tf.zeros_like(x))(tensor)
    for i, axis in enumerate(axes):

        # select restricting indices
        axis_boundaries = boundaries[i, :]
        idx1 = KL.Lambda(lambda x: tf.math.round(tf.random.uniform([1], minval=axis_boundaries[0] * shape[axis],
                                                                   maxval=axis_boundaries[1] * shape[axis])))([])
        idx2 = KL.Lambda(lambda x: tf.math.round(tf.random.uniform([1], minval=axis_boundaries[2] * shape[axis],
                                                                   maxval=axis_boundaries[3] * shape[axis]) - x))(idx1)
        idx3 = KL.Lambda(lambda x: shape[axis] - x[0] - x[1])([idx1, idx2])
        split_idx = KL.Lambda(lambda x: tf.concat([x[0], x[1], x[2]], axis=0))([idx1, idx2, idx3])

        # update mask
        split_list = KL.Lambda(lambda x: tf.split(x[0], tf.cast(x[1], dtype='int32'), axis=axis))([tensor, split_idx])
        tmp_mask = KL.Lambda(lambda x: tf.concat([tf.zeros_like(x[0]), tf.ones_like(x[1]), tf.zeros_like(x[2])],
                                                 axis=axis))([split_list[0], split_list[1], split_list[2]])
        mask = KL.multiply([mask, tmp_mask])

    # mask second_channel
    tensor = KL.multiply([tensor, mask])

    return tensor, mask


# -------------------------------------------------- LABELS TO IMAGE ---------------------------------------------------

def sample_gmm_conditioned_on_labels(labels, means, std_devs, n_labels, n_channels):
    """This function generates an image tensor by sampling a Gaussian Mixture Model conditioned on a label map.
    The generated image can be multi-spectral.
    :param labels: input label map tensor with a batch size of N
    :param means: means of the GMM per channel, should have shape [N, n_labels, n_channels]
    :param std_devs: std devs of the GMM per channel, should have shape [N, n_labels, n_channels]
    :param n_labels: number of labels in the input tensor label map
    :param n_channels: number of channels to generate
    :return: image tensor of shape [N, ..., n_channels]
    """

    # sample from normal distribution
    image = KL.Lambda(lambda x: tf.random.normal(tf.shape(x)))(labels)

    # one channel
    if n_channels == 1:
        means = KL.Lambda(lambda x: K.reshape(tf.split(x, [1, -1])[0], tuple([n_labels])))(means)
        means_map = KL.Lambda(lambda x: tf.gather(x[0], tf.cast(x[1], dtype='int32')))([means, labels])

        std_devs = KL.Lambda(lambda x: K.reshape(tf.split(x, [1, -1])[0], tuple([n_labels])))(std_devs)
        std_devs_map = KL.Lambda(lambda x: tf.gather(x[0], tf.cast(x[1], dtype='int32')))([std_devs, labels])

    # multi-channel
    else:
        cat_labels = KL.Lambda(lambda x: tf.concat([x+n_labels*i for i in range(n_channels)], -1))(labels)

        means = KL.Lambda(lambda x: K.reshape(tf.split(x, [1, -1])[0], tuple([n_labels, n_channels])))(means)
        means = KL.Lambda(lambda x: K.reshape(tf.concat(tf.split(x, [1]*n_channels, axis=-1), 0),
                                              tuple([n_labels*n_channels])))(means)
        means_map = KL.Lambda(lambda x: tf.gather(x[0], tf.cast(x[1], dtype='int32')))([means, cat_labels])

        std_devs = KL.Lambda(lambda x: K.reshape(tf.split(x, [1, -1])[0], tuple([n_labels, n_channels])))(std_devs)
        std_devs = KL.Lambda(lambda x: K.reshape(tf.concat(tf.split(x, [1]*n_channels, axis=-1), 0),
                                                 tuple([n_labels*n_channels])))(std_devs)
        std_devs_map = KL.Lambda(lambda x: tf.gather(x[0], tf.cast(x[1], dtype='int32')))([std_devs, cat_labels])

    # build images based on mean and std maps
    image = KL.multiply([std_devs_map, image])
    image = KL.add([image, means_map])

    return image


# ------------------------------------------------- BLURRING FUNCTIONS -------------------------------------------------

def blur_tensor(tensor, list_kernels, n_dims=3):
    """Blur image with masks in list_kernels, if they are not None."""
    for k in list_kernels:
        if k is not None:
            tensor = KL.Lambda(lambda x: tf.nn.convolution(x[0], x[1], padding='SAME', strides=[1]*n_dims))([tensor, k])
    return tensor


def get_gaussian_1d_kernels(sigma, blurring_range=None):
    """This function builds a list of 1d gaussian blurring kernels.
    The produced tensors are designed to be used with tf.nn.convolution.
    The number of dimensions of the image to blur is assumed to be the length of sigma.
    :param sigma: std deviation of the gaussian kernels to build. Must be a sequence of size n_dims
    (excluding batch and channel dimensions)
    :param blurring_range: if not None, this introduces a randomness in the blurring kernels,
    where sigma is now multiplied by a coefficient dynamically sampled from a uniform distribution with bounds
    [1/blurring_range, blurring_range].
    :return: a list of 1d blurring kernels
    """

    sigma = utils.reformat_to_list(sigma)
    n_dims = len(sigma)

    kernels_list = list()
    for i in range(n_dims):

        if (sigma[i] is None) or (sigma[i] == 0):
            kernels_list.append(None)

        else:
            # build kernel
            if blurring_range is not None:
                random_coef = KL.Lambda(lambda x: tf.random.uniform((1,), 1 / blurring_range, blurring_range))([])
                size = int(math.ceil(2.5 * blurring_range * sigma[i]) / 2)
                kernel = KL.Lambda(lambda x: tfp.distributions.Normal(0., x*sigma[i]).prob(tf.range(start=-size,
                                   limit=size + 1, dtype=tf.float32)))(random_coef)
            else:
                size = int(math.ceil(2.5 * sigma[i]) / 2)
                kernel = KL.Lambda(lambda x: tfp.distributions.Normal(0., sigma[i]).prob(tf.range(start=-size,
                                   limit=size + 1, dtype=tf.float32)))([])
            kernel = KL.Lambda(lambda x: x / tf.reduce_sum(x))(kernel)

            # add dimensions
            for j in range(n_dims):
                if j < i:
                    kernel = KL.Lambda(lambda x: tf.expand_dims(x, 0))(kernel)
                elif j > i:
                    kernel = KL.Lambda(lambda x: tf.expand_dims(x, -1))(kernel)
            kernel = KL.Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, -1), -1))(kernel)  # for tf.nn.convolution
            kernels_list.append(kernel)

    return kernels_list


def blur_channel(tensor, mask, kernels_list, n_dims, blur_background=True):
    """Blur a tensor with a list of kernels.
    If blur_background is True, this function enforces a zero background after blurring in 20% of the cases.
    If blur_background is False, this function corrects edge-blurring effects and replaces the zero-backgound by a low
    intensity gaussian noise.
    :param tensor: a input tensor
    :param mask: mask of non-background regions in the input tensor
    :param kernels_list: list of blurring 1d kernels
    :param n_dims: number of dimensions of the initial image (excluding batch and channel dimensions)
    :param blur_background: whether to correct for edge-blurring effects
    :return: blurred tensor with background augmentation
    """

    # blur image
    tensor = blur_tensor(tensor, kernels_list, n_dims)

    if blur_background:  # background already blurred with the rest of the image

        # enforce zero background in 20% of the cases
        rand = KL.Lambda(lambda x: K.greater(tf.random.uniform((1, 1), 0, 1), 0.8))([])
        tensor = KL.Lambda(lambda y: K.switch(y[0],
                                              KL.Lambda(lambda x: tf.where(tf.cast(x[1], dtype='bool'),
                                                                          x[0], tf.zeros_like(x[0])))([y[1], y[2]]),
                                              y[1]))([rand, tensor, mask])

    else:  # correct for edge blurring effects

        # blur mask and correct edge blurring effects
        blurred_mask = blur_tensor(mask, kernels_list, n_dims)
        tensor = KL.Lambda(lambda x: x[0] / (x[1] + K.epsilon()))([tensor, blurred_mask])

        # replace zero background by low intensity background in 50% of the cases
        rand = KL.Lambda(lambda x: K.greater(tf.random.uniform((1, 1), 0, 1), 0.5))([])
        bckgd_mean = KL.Lambda(lambda x: tf.random.uniform((1, 1), 0, 20))([])
        bckgd_std = KL.Lambda(lambda x: tf.random.uniform((1, 1), 0, 10))([])
        bckgd_mean = KL.Lambda(lambda y: K.switch(y[0],
                                                  KL.Lambda(lambda x: tf.zeros_like(x))(y[1]),
                                                  y[1]))([rand, bckgd_mean])
        bckgd_std = KL.Lambda(lambda y: K.switch(y[0],
                                                 KL.Lambda(lambda x: tf.zeros_like(x))(y[1]),
                                                 y[1]))([rand, bckgd_std])
        background = KL.Lambda(lambda x: x[1] + x[2] * tf.random.normal(tf.shape(x[0])))([tensor, bckgd_mean, bckgd_std])
        background_kernels = get_gaussian_1d_kernels(sigma=[1]*3)
        background = blur_tensor(background, background_kernels, n_dims)
        tensor = KL.Lambda(lambda x: tf.where(tf.cast(x[1], dtype='bool'), x[0], x[2]))([tensor, mask, background])

    return tensor


# ------------------------------------------------ RESAMPLING FUNCTIONS ------------------------------------------------

def resample(tensor,
             resample_shape,
             interp_method='linear',
             subsample_res=None,
             volume_res=None,
             subsample_interp_method='nearest',
             n_dims=3):
    """This function resamples a volume to resample_shape.
    A prior downsampling step can be added if subsample_res is specified. In this case, volume_res should also be
    specified, in order to calculate the downsampling ratio.
    :param tensor: tensor
    :param resample_shape: list or numpy array of size (n_dims,)
    :param interp_method: interpolation method for resampling, 'linear' or 'nearest'
    :param subsample_res: if not None, this triggers a downsampling of the volume, prior to the resampling step.
    list or numpy array of size (n_dims,).
    :param volume_res: if subsample_res is not None, this should be provided to compute downsampling ratio.
     list or numpy array of size (n_dims,).
    :param subsample_interp_method: interpolation method for downsampling, 'linear' or 'nearest'
    :param n_dims: number of dimensions of the initial image (excluding batch and channel dimensions)
    :return: resampled volume
    """

    # downsample image
    downsample_shape = None
    tensor_shape = tensor.get_shape().as_list()[1:-1]
    if subsample_res is not None:
        if subsample_res.tolist() != volume_res.tolist():

            # get shape at which we downsample
            assert volume_res is not None, 'if subsanple_res is specified, so should atlas_res be.'
            downsample_factor = [volume_res[i] / subsample_res[i] for i in range(n_dims)]
            downsample_shape = [int(tensor_shape[i] * downsample_factor[i]) for i in range(n_dims)]

            # downsample volume
            tensor._keras_shape = tuple(tensor.get_shape().as_list())
            tensor = nrn_layers.Resize(size=downsample_shape, interp_method=subsample_interp_method)(tensor)

    # resample image at target resolution
    if resample_shape != downsample_shape:
        tensor._keras_shape = tuple(tensor.get_shape().as_list())
        tensor = nrn_layers.Resize(size=resample_shape, interp_method=interp_method)(tensor)

    return tensor


# ------------------------------------------ INTENSITY AUGMENTATION FUNCTIONS ------------------------------------------

def bias_field_augmentation(tensor, bias_field, n_dims=3):
    """This function takes a bias_field as input, under the form of a small grid.
    The bias field is first resampled to image size, and rescaled to postive values by taking its exponential.
    The bias field is applied by multiplying it with the image."""

    # resize bias field and take exponential
    image_shape = tensor.get_shape().as_list()[1:n_dims + 1]
    bias_field = nrn_layers.Resize(size=image_shape, interp_method='linear')(bias_field)
    bias_field = KL.Lambda(lambda x: K.exp(x))(bias_field)

    # apply bias_field
    tensor._keras_shape = tuple(tensor.get_shape().as_list())
    bias_field._keras_shape = tuple(bias_field.get_shape().as_list())
    return KL.multiply([bias_field, tensor])


def min_max_normalisation(tensor):
    """Normalise tensor between 0 and 1"""
    m = KL.Lambda(lambda x: K.min(x))(tensor)
    M = KL.Lambda(lambda x: K.max(x))(tensor)
    return KL.Lambda(lambda x: (x[0] - x[1]) / (x[2] - x[1]))([tensor, m, M])


def gamma_augmentation(tensor, std=0.5):
    """Raise tensor to a power obtained by taking the exp of a value sampled from a gaussian with specified std dev."""
    return KL.Lambda(lambda x: tf.math.pow(x, tf.math.exp(tf.random.normal([1], mean=0, stddev=std))))(tensor)


# ------------------------------------------------ LABELS EDIT FUNCTIONS -----------------------------------------------

def convert_labels(label_map, labels_list):
    """Change all labels in label_map by the values in labels_list"""
    return KL.Lambda(lambda x: tf.gather(tf.convert_to_tensor(labels_list, dtype='int32'),
                                         tf.cast(x, dtype='int32')))(label_map)


def reset_label_values_to_zero(label_map, labels_to_reset):
    """Reset to zero all occurences in label_map of the values contained in labels_to_remove.
    :param label_map: tensor
    :param labels_to_reset: list of values to reset to zero
    """
    for lab in labels_to_reset:
        label_map = KL.Lambda(lambda x: tf.where(tf.equal(tf.cast(x, dtype='int32'),
                                                          tf.cast(tf.convert_to_tensor(lab), dtype='int32')),
                                                 tf.zeros_like(x, dtype='int32'),
                                                 tf.cast(x, dtype='int32')))(label_map)
    return label_map
