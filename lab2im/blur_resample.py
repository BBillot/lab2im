# python imports
import math
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
import tensorflow_probability as tfp

# project imports
from .utils import utils

# third-party imports
import ext.neuron.layers as nrn_layers


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
        tensor = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
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
        bckgd_mean = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                                  KL.Lambda(lambda x: tf.zeros_like(x))(y[1]),
                                                  y[1]))([rand, bckgd_mean])
        bckgd_std = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                                 KL.Lambda(lambda x: tf.zeros_like(x))(y[1]),
                                                 y[1]))([rand, bckgd_std])
        background = KL.Lambda(lambda x: x[1] + x[2]*tf.random.normal(tf.shape(x[0])))([tensor, bckgd_mean, bckgd_std])
        background_kernels = get_gaussian_1d_kernels(sigma=[1]*3)
        background = blur_tensor(background, background_kernels, n_dims)
        tensor = KL.Lambda(lambda x: tf.where(tf.cast(x[1], dtype='bool'), x[0], x[2]))([tensor, mask, background])

    return tensor


def resample_tensor(tensor,
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
