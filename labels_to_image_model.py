# python imports
import keras
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K

# project imports
import utils
import building_blocks as bb


def labels_to_image_model(labels_shape,
                          n_channels,
                          atlas_res,
                          target_res,
                          data_res,
                          crop_shape,
                          generation_label_list,
                          segmentation_label_list,
                          n_neutral_labels,
                          vox2ras,
                          padding_margin=None,
                          apply_affine_trans=True,
                          apply_nonlin_trans=True,
                          nonlin_shape_factor=0.0625,
                          blur_range=0.15,
                          blur_background=True,
                          thickness=None,
                          downsample=False,
                          apply_bias_field=True,
                          bias_shape_factor=0.025,
                          crop_channel2=None,
                          normalise=True,
                          output_div_by_n=None,
                          flipping=True):
    """
    This function builds a keras/tensorflow model to generate brain images from supplied labels.
    It returns the model as well as the shape ouf the output images without batch and channel dimensions
    (height*width*depth).
    The model takes as inputs:
        -a label image
        -a vector containing the means of the Gaussian distributions to sample for each label,
        -a similar vector for the associated standard deviations.
        -if apply_affine_deformation=True: a (n_dims+1)x(n_dims+1) affine matrix
        -if apply_non_linear_deformation=True: a small non linear field of size batch*x*y*z*n_dims that will be
         resampled to labels size
        -if apply_bias_field=True: a small bias field of size batch*x*y*z*1 that will be resampled to labels size
    The model returns:
        -the generated image
        -the corresponding label map
    :param labels_shape: should be a list or tensor with image dimension plus channel size at the end
    :param n_channels: number of channels to be synthetised
    :param atlas_res: list of dimension resolutions of model's inputs
    :param target_res: list of dimension resolutions of model's outputs
    :param data_res: list of dimension resolutions for image blurring
    :param crop_shape: list, shape of model's outputs
    :param n_neutral_labels: number of non-sided generation labels
    :param generation_label_list: list of all the labels in the dataset (internally converted to [0...N-1] and converted
    back to original values at the end of model)
    :param segmentation_label_list: list of all the labels in the output labels (internally converted to [0...N-1] and
    converted back to original values at the end of model)
    :param vox2ras: example of vox2ras matrix of the labels. Only used to find brain's right/left axis.
    :param padding_margin: margin by which to pad the input labels with zeros. Default is no padding.
    :param apply_affine_trans: whether to apply affine deformation during generation
    :param apply_nonlin_trans: whether to apply non linear deformation during generation
    :param nonlin_shape_factor: if apply_non_linear_deformation=True, factor between the shapes of the labels and of
    the non-linear field that will be sampled
    :param blur_range: Randomise blurring_res. Each element of blurring_res is multiplied at each mini_batch by a random
    coef sampled from a uniform distribution with bounds [1-blur_range, 1+blur_range]. If None, no randomisation.
    :param blur_background: Whether background is a regular label, thus blurred with the others.
    :param downsample: whether to actually downsample the volume image to data_res.
    :param thickness: Size (in mm) of slices (int) or in each dimension (list). (default is None)
    :param apply_bias_field: whether to apply a bias field to the created image during generation
    :param bias_shape_factor: if apply_bias_field=True, factor between the shapes of the labels and of the bias field
    that will be sampled
    :param crop_channel2: stats for cropping second channel along the anterior-posterior axis.
    Should be a vector of length 4, with bounds of uniform distribution for cropping the front and back of the image
    (in percentage). None is no croppping.
    :param normalise: whether to normalise data. Default is False.
    :param output_div_by_n: if not None, make the shape of the output image divisible by this value
    :param flipping: whether to introduce right/left random flipping
    """

    # reformat resolutions
    n_dims, _ = utils.get_dims(labels_shape)
    atlas_res = utils.reformat_to_n_channels_array(atlas_res, n_dims=n_dims)
    if data_res is None:  # data_res assumed to be the same as the atlas
        data_res = atlas_res
    else:
        data_res = utils.reformat_to_n_channels_array(data_res, n_dims=n_dims, n_channels=n_channels)
    atlas_res = atlas_res[0]
    if downsample:  # same as data_res if we want to actually downsample the synthetic image
        downsample_res = data_res
    else:  # set downsample_res to None if downsampling is not necessary
        downsample_res = None
    if target_res is None:
        target_res = atlas_res
    else:
        target_res = utils.reformat_to_n_channels_array(target_res, n_dims)[0]
    thickness = utils.reformat_to_n_channels_array(thickness, n_dims=n_dims, n_channels=n_channels)

    # get shapes
    crop_shape, output_shape, padding_margin = get_shapes(labels_shape, crop_shape, atlas_res, target_res,
                                                          padding_margin, output_div_by_n)

    # create new_label_list and corresponding LUT to make sure that labels go from 0 to N-1
    n_generation_labels = generation_label_list.shape[0]
    new_generation_label_list, lut = utils.rearrange_label_list(generation_label_list)

    # define model inputs
    labels_input = KL.Input(shape=labels_shape+[1], name='labels_input')
    means_input = KL.Input(shape=list(new_generation_label_list.shape) + [n_channels], name='means_input')
    std_devs_input = KL.Input(shape=list(new_generation_label_list.shape) + [n_channels], name='std_devs_input')
    list_inputs = [labels_input, means_input, std_devs_input]
    if apply_affine_trans:
        aff_in = KL.Input(shape=(n_dims + 1, n_dims + 1), name='aff_input')
        list_inputs.append(aff_in)
    else:
        aff_in = None
    if apply_nonlin_trans:
        deformation_field_size = utils.get_resample_shape(labels_shape, nonlin_shape_factor, len(labels_shape))
        nonlin_field_in = KL.Input(shape=deformation_field_size, name='nonlin_input')
        list_inputs.append(nonlin_field_in)
    else:
        nonlin_field_in = None
    if apply_bias_field:
        bias_field_size = utils.get_resample_shape(output_shape, bias_shape_factor, n_channels=1)
        bias_field_in = KL.Input(shape=bias_field_size, name='bias_input')
        list_inputs.append(bias_field_in)
    else:
        bias_field_in = None

    # convert labels to new_label_list
    labels = bb.convert_labels(labels_input, lut)

    # pad labels
    if padding_margin is not None:
        pad = np.transpose(np.array([[0] + padding_margin + [0]] * 2))
        labels = KL.Lambda(lambda x: tf.pad(x, tf.cast(tf.convert_to_tensor(pad), dtype='int32')), name='pad')(labels)
        labels_shape = labels.get_shape().as_list()[1:n_dims+1]

    # deform labels
    if apply_affine_trans | apply_nonlin_trans:
        labels = bb.deform_tensor(labels, aff_in, nonlin_field_in, n_dims)
    labels = KL.Lambda(lambda x: tf.cast(x, dtype='int32'))(labels)

    # cropping
    if crop_shape != labels_shape:
        labels, _ = bb.random_cropping(labels, crop_shape, n_dims)

    if flipping:
        labels, _ = bb.label_map_random_flipping(labels, new_generation_label_list, n_neutral_labels, vox2ras, n_dims)

    # build synthetic image
    image = bb.sample_gmm_conditioned_on_labels(labels, means_input, std_devs_input, n_generation_labels, n_channels)

    # loop over channels
    if n_channels > 1:
        split = KL.Lambda(lambda x: tf.split(x, [1] * n_channels, axis=-1))(image)
    else:
        split = [image]
    mask = KL.Lambda(lambda x: tf.where(tf.greater(x, 0), tf.ones_like(x, dtype='float32'),
                                        tf.zeros_like(x, dtype='float32')))(labels)
    processed_channels = list()
    for i, channel in enumerate(split):

        # reset edges of second channels to zero
        if (crop_channel2 is not None) & (i == 1):  # randomly crop sides of second channel
            channel, tmp_mask = bb.restrict_tensor(channel, axes=3, boundaries=crop_channel2)
        else:
            tmp_mask = None

        # blur channel
        if thickness is not None:
            sigma = utils.get_std_blurring_mask_for_downsampling(data_res[i], atlas_res, thickness=thickness[i])
        else:
            sigma = utils.get_std_blurring_mask_for_downsampling(data_res[i], atlas_res)
        kernels_list = bb.get_gaussian_1d_kernels(sigma, blurring_range=blur_range)
        channel = bb.blur_channel(channel, mask, kernels_list, n_dims, blur_background)
        if (crop_channel2 is not None) & (i == 1):
            channel = KL.multiply([channel, tmp_mask])

        # resample channel
        channel = bb.resample(channel, output_shape, 'linear', downsample_res[i], atlas_res, n_dims=n_dims)

        # apply bias field
        if apply_bias_field:
            channel = bb.bias_field_augmentation(channel, bias_field_in, n_dims=3)

        # intensity augmentation
        channel = KL.Lambda(lambda x: K.clip(x, 0, 300))(channel)
        if normalise:
            channel = bb.min_max_normalisation(channel)
        processed_channels.append(bb.gamma_augmentation(channel, std=0.5))

    # concatenate all channels back
    if n_channels > 1:
        image = KL.concatenate(processed_channels)
    else:
        image = processed_channels[0]

    # resample labels at target resolution
    if crop_shape != output_shape:
        labels = KL.Lambda(lambda x: tf.cast(x, dtype='float32'))(labels)
        labels = bb.resample(labels, output_shape, interp_method='nearest', n_dims=3)
    # convert labels back to original values and reset unwanted labels to zero
    labels = bb.convert_labels(labels, generation_label_list)
    labels_to_reset = [lab for lab in generation_label_list if lab not in segmentation_label_list]
    labels = bb.reset_label_values_to_zero(labels, labels_to_reset)
    labels = KL.Lambda(lambda x: tf.cast(x, dtype='int32'), name='labels_out')(labels)

    # build model (dummy layer enables to keep the labels when plugging this model to other models)
    image = KL.Lambda(lambda x: x[0], name='image_out')([image, labels])
    brain_model = keras.Model(inputs=list_inputs, outputs=[image, labels])
    # shape of returned images
    output_shape = image.get_shape().as_list()[1:]

    return brain_model, output_shape


def get_shapes(labels_shape, output_shape, atlas_res, target_res, padding_margin, output_div_by_n):

    n_dims = len(atlas_res)

    # get new labels shape if padding
    if padding_margin is not None:
        padding_margin = utils.reformat_to_list(padding_margin, length=n_dims, dtype='int')
        labels_shape = [labels_shape[i] + 2 * padding_margin[i] for i in range(n_dims)]

    # get resampling factor
    if atlas_res.tolist() != target_res.tolist():
        resample_factor = [atlas_res[i] / float(target_res[i]) for i in range(n_dims)]
    else:
        resample_factor = None

    # output shape specified, need to get cropping shape, and resample shape if necessary
    if output_shape is not None:
        output_shape = utils.reformat_to_list(output_shape, length=n_dims, dtype='int')

        # make sure that output shape is smaller or equal to label shape
        if resample_factor is not None:
            output_shape = [min(int(labels_shape[i] * resample_factor[i]), output_shape[i]) for i in range(n_dims)]
        else:
            output_shape = [min(labels_shape[i], output_shape[i]) for i in range(n_dims)]

        # make sure output shape is divisible by output_div_by_n
        if output_div_by_n is not None:
            tmp_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n, smaller_ans=True)
                         for s in output_shape]
            if output_shape != tmp_shape:
                print('output shape {0} not divisible by {1}, changed to {2}'.format(output_shape, output_div_by_n,
                                                                                     tmp_shape))
                output_shape = tmp_shape

        # get cropping and resample shape
        if resample_factor is not None:
            cropping_shape = [int(np.around(output_shape[i]/resample_factor[i], 0)) for i in range(n_dims)]
        else:
            cropping_shape = output_shape

    # no output shape specified, so no cropping unless label_shape is not divisible by output_div_by_n
    else:
        cropping_shape = labels_shape
        if resample_factor is not None:
            output_shape = [int(np.around(cropping_shape[i]*resample_factor[i], 0)) for i in range(n_dims)]
        else:
            output_shape = cropping_shape
        # make sure output shape is divisible by output_div_by_n
        if output_div_by_n is not None:
            output_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n, smaller_ans=False)
                            for s in output_shape]

    return cropping_shape, output_shape, padding_margin
