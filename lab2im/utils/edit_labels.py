# python imports
import numpy as np
import keras.layers as KL
from keras.models import Model
from scipy.ndimage import binary_erosion
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import distance_transform_edt

# project imports
from lab2im.utils import utils, edit_volume
from lab2im.blur_resample import get_gaussian_1d_kernels, blur_tensor


def correct_label_map(labels, list_incorrect_labels, list_correct_labels, smooth=False):
    """This function corrects specified label values in a label map by other given values.
    :param labels: a 2d or 3d label map
    :param list_incorrect_labels: list of all label values to correct (e.g. [1, 2, 3, 4]).
    :param list_correct_labels: list of correct label values.
    Correct values must have the same order as their corresponding value in list_incorrect_labels.
    When several correct values are possible for the same incorrect value, the nearest correct value will be selected at
    each voxel to correct. In that case, the different correct values must be specified inside a list whithin
    list_correct_labels (e.g. [10, 20, 30, [40, 50]).
    :param smooth: (optional) whether to smooth the corrected label map
    :return: corrected label map
    """

    # initialisation
    volume_labels = np.unique(labels)
    n_dims, _ = utils.get_dims(labels.shape)
    previous_correct_labels = None
    distance_map_list = None

    # loop over label values
    for incorrect_label, correct_label in zip(list_incorrect_labels, list_correct_labels):
        if incorrect_label in volume_labels:

            # only one possible value to replace with
            if isinstance(correct_label, (int, float, np.int64, np.int32, np.int16, np.int8)):
                incorrect_voxels = np.where(labels == incorrect_label)
                labels[incorrect_voxels] = correct_label

            # several possibilities
            elif isinstance(correct_label, (tuple, list)):
                mask = np.zeros(labels.shape, dtype='bool')

                # crop around label to correct
                for lab in correct_label:
                    mask = mask | (labels == lab)
                _, cropping = edit_volume.crop_volume_around_region(mask, margin=10)
                if n_dims == 2:
                    tmp_im = labels[cropping[0]:cropping[2], cropping[1]:cropping[3], ...]
                elif n_dims == 3:
                    tmp_im = labels[cropping[0]:cropping[3], cropping[1]:cropping[4], cropping[2]:cropping[5], ...]
                else:
                    raise ValueError('cannot correct volumes with more than 3 dimensions')

                # calculate distance maps for all new label candidates
                incorrect_voxels = np.where(tmp_im == incorrect_label)
                if correct_label != previous_correct_labels:
                    distance_map_list = [distance_transform_edt(np.logical_not(tmp_im == lab))
                                         for lab in correct_label]
                    previous_correct_labels = correct_label
                distances_correct = np.stack([dist[incorrect_voxels] for dist in distance_map_list])

                # select nearest value
                idx_correct_lab = np.argmin(distances_correct, axis=0)
                tmp_im[incorrect_voxels] = np.array(correct_label)[idx_correct_lab]
                if n_dims == 2:
                    labels[cropping[0]:cropping[2], cropping[1]:cropping[3], ...] = tmp_im
                else:
                    labels[cropping[0]:cropping[3], cropping[1]:cropping[4], cropping[2]:cropping[5], ...] = tmp_im

    # smoothing
    if smooth:
        kernel = np.ones(tuple([3] * n_dims))
        labels = smooth_label_map(labels, kernel)

    return labels


def mask_label_map(labels, masking_values, masking_value=0, return_mask=False):
    """
    This function masks a label map around a list of specified values.
    :param labels: input label map
    :param masking_values: list of values to mask around
    :param masking_value: (optional) value to mask the label map with
    :param return_mask: (optional) whether to return the applied mask
    :return: the masked label map, and the applied mask if return_mask is True.
    """

    # build mask and mask labels
    mask = np.zeros(labels.shape, dtype=bool)
    masked_labels = labels.copy()
    for value in masking_values:
        mask = mask | (labels == value)
    masked_labels[np.logical_not(mask)] = masking_value

    if return_mask:
        mask = mask * 1
        return masked_labels, mask
    else:
        return masked_labels


def smooth_label_map(labels, kernel, print_progress=0):
    """This function smooth an input label map by replacing each voxel by the value of its most numerous neigbour.
    :param labels: input label map
    :param kernel: kernel when counting neighbours. Must contain only zeros or ones.
    :param print_progress: (optional) If not 0, interval at which to print the number of processed labels.
    :return: smoothed label map
    """
    # get info
    labels_shape = labels.shape
    label_list = np.unique(labels).astype('int32')

    # loop through label values
    count = np.zeros(labels_shape)
    labels_smoothed = np.zeros(labels_shape, dtype='int')
    for la, label in enumerate(label_list):
        if print_progress:
            utils.print_loop_info(la, len(label_list), print_progress)

        # count neigbours with same value
        mask = (labels == label) * 1
        n_neighbours = convolve(mask, kernel)

        # update label map and maximum neigbour counts
        idx = n_neighbours > count
        count[idx] = n_neighbours[idx]
        labels_smoothed[idx] = label

    return labels_smoothed


def erode_label_map(labels, labels_to_erode, erosion_factors=1, gpu=False, model=None, return_model=False):
    """Erode a given set of label values within a label map.
    :param labels: a 2d or 3d label map
    :param labels_to_erode: list of label values to erode
    :param erosion_factors: (optional) list of erosion factors to use for each label. If values are integers, normal
    erosion applies. If float, we first 1) blur a mask of the corresponding label value, and 2) use the erosion factor
    as a threshold in the blurred mask.
    If erosion_factors is a single value, the same factor will be applied to all labels.
    :param gpu: (optionnal) whether to use a fast gpu model for blurring (if erosion factors are floats)
    :param model: (optionnal) gpu model for blurring masks (if erosion factors are floats)
    :param return_model: (optional) whether to return the gpu blurring model
    :return: eroded label map, and gpu blurring model is return_model is True.
    """
    # reformat labels_to_erode and erode
    labels_to_erode = utils.reformat_to_list(labels_to_erode)
    erosion_factors = utils.reformat_to_list(erosion_factors, length=len(labels_to_erode))
    labels_shape = list(labels.shape)
    n_dims, _ = utils.get_dims(labels_shape)

    # loop over labels to erode
    for label_to_erode, erosion_factor in zip(labels_to_erode, erosion_factors):

        assert erosion_factor > 0, 'all erosion factors should be strictly positive, had {}'.format(erosion_factor)

        # get mask of current label value
        mask = (labels == label_to_erode)

        # erode as usual if erosion factor is int
        if int(erosion_factor) == erosion_factor:
            erode_struct = utils.build_binary_structure(int(erosion_factor), n_dims)
            eroded_mask = binary_erosion(mask, erode_struct)

        # blur mask and use erosion factor as a threshold if float
        else:
            if gpu:
                if model is None:
                    mask_in = KL.Input(shape=labels_shape + [1], dtype='float32')
                    list_k = get_gaussian_1d_kernels([1] * 3)
                    blurred_mask = blur_tensor(mask_in, list_k, n_dims=n_dims)
                    model = Model(inputs=mask_in, outputs=blurred_mask)
                eroded_mask = model.predict(utils.add_axis(np.float32(mask), -2))
            else:
                eroded_mask = edit_volume.blur_volume(np.float32(mask), 1)
            eroded_mask = np.squeeze(eroded_mask) > erosion_factor

        # crop label map and mask around values to change
        mask = mask & np.logical_not(eroded_mask)
        cropped_lab_mask, cropping = edit_volume.crop_volume_around_region(mask, margin=3)
        croppped_labels = edit_volume.crop_volume_with_idx(labels, cropping)

        # calculate distance maps for all labels in cropped_labels
        labels_list = np.unique(croppped_labels)
        labels_list = labels_list[labels_list != label_to_erode]
        list_dist_maps = [distance_transform_edt(np.logical_not(croppped_labels == la)) for la in labels_list]
        candidate_distances = np.stack([dist[cropped_lab_mask] for dist in list_dist_maps])

        # select nearest value and put cropped labels back to full label map
        idx_correct_lab = np.argmin(candidate_distances, axis=0)
        croppped_labels[cropped_lab_mask] = np.array(labels_list)[idx_correct_lab]
        if n_dims == 2:
            labels[cropping[0]:cropping[2], cropping[1]:cropping[3], ...] = croppped_labels
        elif n_dims == 3:
            labels[cropping[0]:cropping[3], cropping[1]:cropping[4], cropping[2]:cropping[5], ...] = croppped_labels

        if return_model:
            return labels, model
        else:
            return labels


def compute_hard_volumes(labels, voxel_volume=1., label_list=None, skip_background=True):
    """Compute hard volumes in a label map.
    :param labels: a label map
    :param voxel_volume: (optional) volume of voxel. Default is 1 (i.e. returned volumes are voxel counts).
    :param label_list: (optional) list of labels to compute volumes for. Can be an int, a sequence, or a numpy array.
    If None, the volumes of all label values are computed.
    :param skip_background: (optional) whether to skip computing the volume of the background.
    If label_list is None, this assumes background value is 0.
    If label_list is not None, this assumes the background is the first value in label list.
    :return: numpy 1d vector with the volumes of each structure
    """

    # initialisation
    subject_label_list = utils.reformat_to_list(np.unique(labels), dtype='int')
    if label_list is None:
        label_list = subject_label_list
    else:
        label_list = utils.reformat_to_list(label_list)
    if skip_background:
        label_list = label_list[1:]
    volumes = np.zeros(len(label_list))

    # loop over label values
    for idx, label in enumerate(label_list):
        if label in subject_label_list:
            mask = (labels == label) * 1
            volumes[idx] = np.sum(mask)
        else:
            volumes[idx] = 0

    return volumes * voxel_volume
