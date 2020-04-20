# python imports
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

# project imports
from lab2im.utils import utils, edit_labels


def mask_volume(volume, mask=None, threshold=0.1, dilate=0, erode=0, masking_value=0, return_mask=False):
    """Mask a volume, either with a given mask, or by keeping only the values above a threshold.
    :param volume: a numpy array, possibly with several channels
    :param mask: (optional) a numpy array to mask volume with.
    Mask doesn't have to be a 0/1 array, all strictly positive values of mask are considered for masking volume.
    Mask should have the same size as volume. If volume has several channels, mask can either be uni- or multi-channel.
     In the first case, the same mask is applied to all channels.
    :param threshold: (optional) If mask is None, masking is performed by keeping thresholding the input.
    :param dilate: (optional) number of voxels by which to dilate the provided or computed mask.
    :param erode: (optional) number of voxels by which to erode the provided or computed mask.
    :param masking_value: (optional) masking value
    :param return_mask: (optional) whether to return the applied mask
    :return: the masked volume, and the applied mask if return_mask is True.
    """

    vol_shape = list(volume.shape)
    n_dims, n_channels = utils.get_dims(vol_shape)

    # get mask and erode/dilate it
    if mask is None:
        mask = volume >= threshold
    else:
        assert list(mask.shape[:n_dims]) == vol_shape[:n_dims], 'mask should have shape {0}, or {1}, had {2}'.format(
            vol_shape[:n_dims], vol_shape[:n_dims] + [n_channels], list(mask.shape))
        mask = mask > 0
    if dilate > 0:
        dilate_struct = utils.build_binary_structure(dilate, n_dims)
        mask_to_apply = binary_dilation(mask, dilate_struct)
    else:
        mask_to_apply = mask
    if erode > 0:
        erode_struct = utils.build_binary_structure(erode, n_dims)
        mask_to_apply = binary_erosion(mask_to_apply, erode_struct)
    mask_to_apply = mask | mask_to_apply

    # replace values outside of mask by padding_char
    if mask_to_apply.shape == volume.shape:
        volume[np.logical_not(mask_to_apply)] = masking_value
    else:
        volume[np.stack([np.logical_not(mask_to_apply)] * n_channels, axis=-1)] = masking_value

    if return_mask:
        return volume, mask_to_apply
    else:
        return volume


def rescale_volume(volume, new_min=0, new_max=255, min_percentile=0.025, max_percentile=0.975, use_positive_only=True):
    """This function linearly rescales a volume between new_min and new_max.
    :param volume: a numpy array
    :param new_min: (optional) minimum value for the rescaled image.
    :param new_max: (optional) maximum value for the rescaled image.
    :param min_percentile: (optional) percentile for estimating robust minimum of volume
    :param max_percentile: (optional) percentile for estimating robust maximum of volume
    :param use_positive_only: (optional) whether to use only positive values when estimating the min and max percentile
    :return: rescaled volume
    """

    # sort intensities
    if use_positive_only:
        intensities = np.sort(volume[volume > 0])
    else:
        intensities = np.sort(volume.flatten())

    # define robust max and min
    robust_min = np.maximum(0, intensities[int(intensities.shape[0] * min_percentile)])
    robust_max = intensities[int(intensities.shape[0] * max_percentile)]

    # trim values outside range
    volume = np.clip(volume, robust_min, robust_max)

    # rescale image
    volume = new_min + (volume-robust_min) / (robust_max-robust_min) * new_max

    return volume


def crop_volume(volume, cropping_margin=None, cropping_shape=None, aff=None):
    """Crop volume by a given margin, or to a given shape.
    :param volume: 2d or 3d numpy array (possibly with multiple channels)
    :param cropping_margin: (optional) margin by which to crop the volume. Can be an int, sequence or 1d numpy array of
    size n_dims. Should be given if cropping_shape is None.
    :param cropping_shape: (optional) shape to which the volume will be cropped. Can be an int, sequence or 1d numpy
    array of size n_dims. Should be given if cropping_margin is None.
    :param aff: (optional) affine matrix of the input volume.
    If not None, this function also returns an updated version of the affine matrix for the cropped volume.
    :return: cropped volume, and corresponding affine matrix if aff is not None.
    """

    assert (cropping_margin is not None) | (cropping_shape is not None), \
        'cropping_margin or cropping_shape should be provided'
    assert not ((cropping_margin is not None) & (cropping_shape is not None)), \
        'only one of cropping_margin or cropping_shape should be provided'

    # get info
    vol_shape = volume.shape
    n_dims, _ = utils.get_dims(vol_shape)

    # find cropping indices
    if cropping_margin is not None:
        cropping_margin = utils.reformat_to_list(cropping_margin, length=n_dims)
        min_crop_idx = cropping_margin
        max_crop_idx = [vol_shape[i] - cropping_margin[i] for i in range(n_dims)]
        assert (np.array(max_crop_idx) >= np.array(min_crop_idx)).all(), 'cropping_margin is larger than volume shape'
    else:
        cropping_shape = utils.reformat_to_list(cropping_shape, length=n_dims)
        min_crop_idx = [int((vol_shape[i] - cropping_shape[i]) / 2) for i in range(n_dims)]
        max_crop_idx = [min_crop_idx[i] + cropping_shape[i] for i in range(n_dims)]
        assert (np.array(min_crop_idx) >= 0).all(), 'cropping_shape is larger than volume shape'
    crop_idx = np.concatenate([np.array(min_crop_idx), np.array(max_crop_idx)])

    # crop volume
    if n_dims == 2:
        volume = volume[crop_idx[0]: crop_idx[2], crop_idx[1]: crop_idx[3], ...]
    elif n_dims == 3:
        volume = volume[crop_idx[0]: crop_idx[3], crop_idx[1]: crop_idx[4], crop_idx[2]: crop_idx[5], ...]

    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ np.array(min_crop_idx)
        return volume, aff
    else:
        return volume


def crop_volume_around_region(volume, mask=None, threshold=0.1, masking_labels=None, margin=0, aff=None):
    """Crop a volume around a specific region. This region is defined by a mask obtained by either
    1) directly specifying it as input
    2) thresholding the input volume
    3) keeping a set of label values if the volume is a label map.
    :param volume: a 2d or 3d numpy array
    :param mask: (optional) mask of region to crop around. Must be same size as volume. Can either be boolean or 0/1.
    :param threshold: (optional) if mask is None, lower bound to determine values to crop around
    :param masking_labels: (optional) if mask is None, and if the volume is a label map, it can be cropped around a
    set of labels specified in masking_labels, which can either be a single int, a sequence or a 1d numpy array.
    :param margin: (optional) add margin around mask
    :param aff: (optional) if specified, this function returns an updated affine matrix of the volume after cropping.
    :return: the cropped volume, the cropping indices (in the order [lower_bound_dim_1, ..., upper_bound_dim_1, ...]),
    and the updated affine matrix if aff is not None.
    """

    n_dims, _ = utils.get_dims(volume.shape)

    # mask ROIs for cropping
    if mask is None:
        if masking_labels is not None:
            masked_volume, mask = edit_labels.mask_label_map(volume, masking_values=masking_labels, return_mask=True)
        else:
            mask = volume > threshold

    # find cropping indices
    indices = np.nonzero(mask)
    min_idx = np.maximum(np.array([np.min(idx) for idx in indices]) - margin, 0)
    max_idx = np.minimum(np.array([np.max(idx) for idx in indices]) + margin, np.array(volume.shape))
    cropping = np.concatenate([min_idx, max_idx])

    # crop volume
    if n_dims == 3:
        volume = volume[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2], ...]
    elif n_dims == 2:
        volume = volume[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], ...]
    else:
        raise ValueError('cannot crop volumes with more than 3 dimensions')

    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ min_idx
        return volume, cropping, aff
    else:
        return volume, cropping


def crop_volume_with_idx(volume, crop_idx, aff=None):
    """Crop a volume with given indices.
    :param volume: a 2d or 3d numpy array
    :param crop_idx: croppping indices, in the order [lower_bound_dim_1, ..., upper_bound_dim_1, ...].
    Can be a list or a 1d numpy array.
    :param aff: (optional) if specified, this function returns an updated affine matrix of the volume after cropping.
    :return: the cropped volume, and the updated affine matrix if aff is not None.
    """

    # crop image
    n_dims = int(crop_idx.shape[0] / 2)
    if n_dims == 2:
        volume = volume[crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3], ...]
    elif n_dims == 3:
        volume = volume[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], ...]
    else:
        raise Exception('cannot crop volumes with more than 3 dimensions')

    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ crop_idx[:3]
        return volume, aff
    else:
        return volume


def pad_volume(volume, padding_shape, padding_value=0, aff=None):
    """Pad volume to a given shape
    :param volume: volume to be padded
    :param padding_shape: shape to pad volume to. Can be a number, a sequence or a 1d numpy array.
    :param padding_value: (optional) value used for padding
    :param aff: (optional) affine matrix of the volume
    :return: padded volume, and updated affine matrix if aff is not None.
    """
    # get info
    vol_shape = volume.shape
    n_dims, _ = utils.get_dims(vol_shape)
    n_channels = len(vol_shape) - len(vol_shape[:n_dims])
    padding_shape = utils.reformat_to_list(padding_shape, length=n_dims, dtype='int')

    # get padding margins
    min_pad_margins = np.int32(np.floor((np.array(padding_shape) - np.array(vol_shape)) / 2))
    max_pad_margins = np.int32(np.ceil((np.array(padding_shape) - np.array(vol_shape)) / 2))
    if (min_pad_margins < 0).any():
        raise ValueError('volume is bigger than provided shape')
    pad_margins = tuple([(min_pad_margins[i], max_pad_margins[i]) for i in range(n_dims)])
    if n_channels > 1:
        pad_margins += [[0, 0]]

    # pad volume
    volume = np.pad(volume, pad_margins, mode='constant', constant_values=padding_value)

    if aff is not None:
        aff[:-1, -1] = aff[:-1, -1] - aff[:-1, :-1] @ min_pad_margins
        return volume, aff
    else:
        return volume


def flip_volume(volume, axis=None, direction=None, aff=None):
    """Flip volume along a specified axis.
    If unknown, this axis can be inferred from an affine matrix with a specified anatomical direction.
    :param volume: a numpy array
    :param axis: (optional) axis along which to flip the volume. Can either be an int or a tuple.
    :param direction: (optional) if axis is None, the volume can be flipped along an anatomical direction:
    'rl' (right/left), 'ap' anterior/posterior), 'si' (superior/inferior).
    :param aff: (optional) please provide an affine matrix if direction is not None
    :return: flipped volume
    """

    assert (axis is not None) | ((aff is not None) & (direction is not None)), \
        'please provide either axis, or an affine matrix with a direction'

    # get flipping axis from aff if axis not provided
    if (axis is None) & (aff is not None):
        volume_axes, _ = get_ras_axes_and_signs(aff)
        if direction == 'rl':
            axis = volume_axes[0]
        elif direction == 'ap':
            axis = volume_axes[1]
        elif direction == 'si':
            axis = volume_axes[2]
        else:
            raise ValueError("direction should be 'rl', 'ap', or 'si', had %s" % direction)

    # flip volume
    return np.flip(volume, axis=axis)


def get_ras_axes_and_signs(aff, n_dims=3):
    """This function finds the RAS axes corresponding to each dimension of a volume, based on its affine matrix.
    :param aff: affine matrix Can be a 2d numpy array of size n_dims*n_dims, n_dims+1*n_dims+1, or n_dims*n_dims+1.
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: two numpy 1d arrays of lengtn n_dims, one with the axes corresponding to RAS orientations,
    and one with their corresponding direction.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    img_ras_signs = np.sign(aff_inverted[img_ras_axes, np.arange(3)])
    return img_ras_axes, img_ras_signs


def align_volume_to_ref(volume, ref_ras_axes, ref_ras_signs, vol_ras_axes, vol_ras_signs):
    """This function aligns a volume to a given orientation (axis and direction).
    :param volume: a numpy array
    :param ref_ras_axes: ras axes along which to align volume
    :param ref_ras_signs: ras directions along which to align volume
    :param vol_ras_axes: current ras axes
    :param vol_ras_signs: current ras signs
    :return: aligned volume
    """

    n_dims, _ = utils.get_dims(volume.shape)

    # work on copies
    cp_ref_ras_axes = ref_ras_axes.copy()
    cp_ref_ras_signs = ref_ras_signs.copy()
    cp_im_ras_axes = vol_ras_axes.copy()
    cp_im_ras_signs = vol_ras_signs.copy()

    # align axis
    for i in range(n_dims):
        volume = np.swapaxes(volume, cp_im_ras_axes[i], cp_ref_ras_axes[i])
        swapped_axis_idx = np.where(cp_im_ras_axes == cp_ref_ras_axes[i])
        cp_im_ras_axes[swapped_axis_idx], cp_im_ras_axes[i] = cp_im_ras_axes[i], cp_im_ras_axes[swapped_axis_idx]
        cp_im_ras_signs[swapped_axis_idx], cp_im_ras_signs[i] = cp_im_ras_signs[i], cp_im_ras_signs[swapped_axis_idx]

    # align directions
    axis_to_align = tuple(ref_ras_axes[cp_im_ras_signs != cp_ref_ras_signs])
    volume = np.flip(volume, axis=axis_to_align)
    return volume


def blur_volume(volume, sigma, mask=None):
    """Blur volume with gaussian masks of given sigma.
    :param volume: 2d or 3d numpy array
    :param sigma: standard deviation of the gaussian kernels. Can be a number, a sequence or a 1d numpy array
    :param mask: (optional) numpy array of the same shape as volume to correct for edge blurring effects.
    Mask can be a boolean or numerical array. In the later, the mask is computed by keeping all values above zero.
    :return: blurred volume
    """

    # initialisation
    n_dims, _ = utils.get_dims(volume.shape)
    sigma = utils.reformat_to_list(sigma, length=n_dims, dtype='float')

    # blur image
    volume = gaussian_filter(volume, sigma=sigma, mode='nearest')  # nearest refers to edge padding

    # correct edge effect if mask is not None
    if mask is not None:
        assert volume.shape == mask.shape, 'volume and mask should have the same dimensions: ' \
                                           'got {0} and {1}'.format(volume.shape, mask.shape)
        mask = (mask > 0) * 1.0
        blurred_mask = gaussian_filter(mask, sigma=sigma, mode='nearest')
        volume = volume / (blurred_mask + 1e-6)
        volume[mask == 0] = 0

    return volume
