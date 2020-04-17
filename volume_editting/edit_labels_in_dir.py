# python imports
import os
import csv
import numpy as np
import tensorflow as tf
import keras.layers as KL
from keras.models import Model

# project imports
import utils
import building_blocks
from volume_editting import edit_labels


def correct_labels_in_dir(labels_dir, list_incorrect_labels, list_correct_labels, results_dir, smooth=False,
                          recompute=True):
    """This function corrects label values for all labels in a folder.
    :param labels_dir: path of directory with input label maps
    :param list_incorrect_labels: list of all label values to correct (e.g. [1, 2, 3, 4]).
    :param list_correct_labels: list of correct label values.
    Correct values must have the same order as their corresponding value in list_incorrect_labels.
    When several correct values are possible for the same incorrect value, the nearest correct value will be selected at
    each voxel to correct. In that case, the different correct values must be specified inside a list whithin
    list_correct_labels (e.g. [10, 20, 30, [40, 50]).
    :param results_dir: path of directory where corrected label maps will be writen
    :param smooth: (optional) whether to smooth the corrected label maps
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # prepare data files
    path_labels = utils.list_images_in_folder(labels_dir)
    for idx, path_label in enumerate(path_labels):
        utils.print_loop_info(idx, len(path_labels), 10)

        # correct labels
        path_result = os.path.join(results_dir, os.path.basename(path_label))
        if (not os.path.isfile(path_result)) | recompute:
            im, vox2ras, header = utils.load_volume(path_label, im_only=False)
            im = edit_labels.correct_label_map(im, list_incorrect_labels, list_correct_labels, smooth=smooth)
            utils.save_volume(im, vox2ras, header, path_result)


def mask_labels_in_dir(labels_dir, result_dir, values_to_keep, masking_value=0, mask_result_dir=None, recompute=True):
    """This function masks all label maps in a folder by keeping a set of given label values.
    :param labels_dir: path of directory with input label maps
    :param result_dir: path of directory where corrected label maps will be writen
    :param values_to_keep: list of values for masking the label maps.
    :param masking_value: (optional) value to mask the label maps with
    :param mask_result_dir: (optional) path of directory where applied masks will be writen
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    if mask_result_dir is not None:
        if not os.path.isdir(mask_result_dir):
            os.mkdir(mask_result_dir)

    # reformat values to keep
    if isinstance(values_to_keep, (int, float)):
        values_to_keep = [values_to_keep]
    elif not isinstance(values_to_keep, (tuple, list)):
        raise TypeError('values to keep should be int, float, tuple, or list')

    # loop over labels
    path_labels = utils.list_images_in_folder(labels_dir)
    for idx, path_label in enumerate(path_labels):
        utils.print_loop_info(idx, len(path_labels), 10)

        # mask labels
        path_result = os.path.join(result_dir, os.path.basename(path_label))
        if mask_result_dir is not None:
            path_result_mask = os.path.join(mask_result_dir, os.path.basename(path_label))
        else:
            path_result_mask = ''
        if (not os.path.isfile(path_result)) | \
                (mask_result_dir is not None) & (not os.path.isfile(path_result_mask)) | \
                recompute:
            lab, aff, h = utils.load_volume(path_label, im_only=False)
            if mask_result_dir is not None:
                labels, mask = edit_labels.mask_label_map(lab, values_to_keep, masking_value, return_mask=True)
                path_result_mask = os.path.join(mask_result_dir, os.path.basename(path_label))
                utils.save_volume(mask, aff, h, path_result_mask)
            else:
                labels = edit_labels.mask_label_map(lab, values_to_keep, masking_value, return_mask=False)
            utils.save_volume(labels, aff, h, path_result)


def smooth_labels_in_dir(labels_dir, result_dir, gpu=False, path_label_list=None, recompute=True):
    """Smooth all label maps in a folder by replacing each voxel by the value of its most numerous neigbours.
    :param labels_dir: path of directory with input label maps
    :param result_dir: path of directory where smoothed label maps will be writen
    :param gpu: (optional) whether to use a gpu implementation for faster processing
    :param path_label_list: (optionnal) if gpu is True, path of numpy array with all label values.
    Automatically computed if not provided.
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # list label maps
    path_labels = utils.list_images_in_folder(labels_dir)

    if gpu:
        # initialisation
        label_list, _ = utils.get_list_labels(path_label_list=path_label_list, labels_dir=labels_dir, FS_sort=True)
        previous_model_input_shape = None
        smoothing_model = None

        # loop over label maps
        for idx, path_label in enumerate(path_labels):
            utils.print_loop_info(idx, len(path_labels), 10)

            # smooth label map
            path_result = os.path.join(result_dir, os.path.basename(path_label))
            if (not os.path.isfile(path_result)) | recompute:
                labels, label_shape, aff, n_dims, _, h, _ = utils.get_volume_info(path_label, return_volume=True)
                if label_shape != previous_model_input_shape:
                    previous_model_input_shape = label_shape
                    smoothing_model = smoothing_gpu_model(label_shape, label_list)
                labels = smoothing_model.predict(utils.add_axis(labels))
                utils.save_volume(np.squeeze(labels), aff, h, path_result, dtype='int')

    else:
        # build kernel
        _, _, n_dims, _, _, _ = utils.get_volume_info(path_labels[0])
        kernel = np.ones(tuple([3] * n_dims))

        # loop over label maps
        for idx, path in enumerate(path_labels):
            utils.print_loop_info(idx, len(path_labels), 10)

            # smooth label map
            path_result = os.path.join(result_dir, os.path.basename(path))
            if (not os.path.isfile(path_result)) | recompute:
                volume, aff, h = utils.load_volume(path, im_only=False)
                new_volume = edit_labels.smooth_label_map(volume, kernel)
                utils.save_volume(new_volume, aff, h, path_result, dtype='int')


def smoothing_gpu_model(label_shape, label_list):
    """This function builds a gpu model in keras with a tensorflow backend to smooth label maps.
    This model replaces each voxel of the input by the value of its most numerous neigbour.
    :param label_shape: shape of the label map
    :param label_list: list of all labels to consider
    :return: gpu smoothing model
    """

    # create new_label_list and corresponding LUT to make sure that labels go from 0 to N-1
    n_labels = label_list.shape[0]
    _, lut = utils.rearrange_label_list(label_list)

    # convert labels to new_label_list and use one hot encoding
    labels_in = KL.Input(shape=label_shape, name='lab_input', dtype='int32')
    labels = building_blocks.convert_labels(labels_in, lut)
    labels = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, dtype='int32'), depth=n_labels, axis=-1))(labels)

    # count neighbouring voxels
    n_dims, _ = utils.get_dims(label_shape)
    kernel = KL.Lambda(lambda x: tf.convert_to_tensor(
        utils.add_axis(utils.add_axis(np.ones(tuple([n_dims]*n_dims)).astype('float32'), -1), -1)))([])
    split = KL.Lambda(lambda x: tf.split(x, [1] * n_labels, axis=-1))(labels)
    labels = KL.Lambda(lambda x: tf.nn.convolution(x[0], x[1], padding='SAME'))([split[0], kernel])
    for i in range(1, n_labels):
        tmp = KL.Lambda(lambda x: tf.nn.convolution(x[0], x[1], padding='SAME'))([split[i], kernel])
        labels = KL.Lambda(lambda x: tf.concat([x[0], x[1]], -1))([labels, tmp])

    # take the argmax and convert labels to original values
    labels = KL.Lambda(lambda x: tf.math.argmax(x, -1))(labels)
    labels = building_blocks.convert_labels(labels, label_list)
    return Model(inputs=labels_in, outputs=labels)


def erode_labels_in_dir(labels_dir, result_dir, labels_to_erode, erosion_factors=1., gpu=False, recompute=True):
    """Erode a given set of label values for all label maps in a folder.
    :param labels_dir: path of directory with input label maps
    :param result_dir: path of directory where cropped label maps will be writen
    :param labels_to_erode: list of label values to erode
    :param erosion_factors: (optional) list of erosion factors to use for each label value. If values are integers,
    normal erosion applies. If float, we first 1) blur a mask of the corresponding label value with a gpu model,
    and 2) use the erosion factor as a threshold in the blurred mask.
    If erosion_factors is a single value, the same factor will be applied to all labels.
    :param gpu: (optionnal) whether to use a fast gpu model for blurring (if erosion factors are floats)
    :param recompute: (optional) whether to recompute result files even if they already exists
    """
    # create result dir
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # loop over label maps
    model = None
    path_labels = utils.list_images_in_folder(labels_dir)
    for idx, path_label in enumerate(path_labels):
        utils.print_loop_info(idx, len(path_labels), 5)

        # erode label map
        labels, aff, h = utils.load_volume(path_label, im_only=False)
        path_result = os.path.join(result_dir, os.path.basename(path_label))
        if (not os.path.isfile(path_result)) | recompute:
            labels, model = edit_labels.erode_label_map(labels, labels_to_erode, erosion_factors, gpu, model,
                                                        return_model=True)
            utils.save_volume(labels, aff, h, path_result)


def upsample_labels_in_dir(labels_dir,
                           target_res,
                           result_dir,
                           path_label_list=None,
                           path_freesurfer='/usr/local/freesurfer/',
                           recompute=True):
    """This funtion upsamples all label maps within a folder. Importantly, each label map is converted into probability
    maps for all label values, and all these maps are upsampled separetely. The upsampled label maps are recovered by
    taking the argmax of the label values probability maps.
    :param labels_dir: path of directory with label maps to upsample
    :param target_res: resolution at which to upsample the label maps. can be a single number (isotropic), or a list.
    :param result_dir: path of directory where the upsampled label maps will be writen
    :param path_label_list: (optional) path of numpy array containing all label values.
    Computed automatically if not given.
    :param path_freesurfer: (optional) path freesurfer home (upsampling performed with mri_convert)
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # prepare result dir
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # set up FreeSurfer
    os.environ['FREESURFER_HOME'] = path_freesurfer
    os.system(os.path.join(path_freesurfer, 'SetUpFreeSurfer.sh'))
    mri_convert = os.path.join(path_freesurfer, 'bin/mri_convert.bin')

    # list label maps
    path_labels = utils.list_images_in_folder(labels_dir)
    labels_shape, aff, n_dims, _, h, _ = utils.get_volume_info(path_labels[0])

    # build command
    target_res = utils.reformat_to_list(target_res, length=n_dims)
    post_cmd = ' -voxsize ' + ' '.join([str(r) for r in target_res]) + ' -odt float'

    # load label list and corresponding LUT to make sure that labels go from 0 to N-1
    label_list, _ = utils.get_list_labels(path_label_list, labels_dir=path_labels, FS_sort=True)
    new_label_list, lut = utils.rearrange_label_list(label_list)

    # loop over label maps
    for idx, path_label in enumerate(path_labels):
        utils.print_loop_info(idx, len(path_labels), 5)
        path_result = os.path.join(result_dir, os.path.basename(path_label))
        if (not os.path.isfile(path_result)) | recompute:

            # load volume
            labels, aff, h = utils.load_volume(path_label, im_only=False)
            labels = lut[labels.astype('int')]

            # create individual folders for label map
            basefilename = utils.strip_extension(os.path.basename(path_label))
            indiv_label_dir = os.path.join(result_dir, basefilename)
            upsample_indiv_label_dir = os.path.join(result_dir, basefilename + '_upsampled')
            if not os.path.isdir(indiv_label_dir):
                os.mkdir(indiv_label_dir)
            if not os.path.isdir(upsample_indiv_label_dir):
                os.mkdir(upsample_indiv_label_dir)

            # loop over label values
            for label in new_label_list:
                path_mask = os.path.join(indiv_label_dir, str(label)+'.nii.gz')
                path_mask_upsampled = os.path.join(upsample_indiv_label_dir, str(label)+'.nii.gz')
                if not os.path.isfile(path_mask):
                    mask = (labels == label) * 1.0
                    utils.save_volume(mask, aff, h, path_mask)
                if not os.path.isfile(path_mask_upsampled):
                    cmd = mri_convert + ' ' + path_mask + ' ' + path_mask_upsampled + post_cmd
                    os.system(cmd)

            # compute argmax of upsampled probability maps (upload them one at a time)
            probmax, aff, h = utils.load_volume(os.path.join(upsample_indiv_label_dir, '0.nii.gz'), im_only=False)
            labels = np.zeros(probmax.shape, dtype='int')
            for label in new_label_list:
                prob = utils.load_volume(os.path.join(upsample_indiv_label_dir, str(label) + '.nii.gz'))
                idx = prob > probmax
                labels[idx] = label
                probmax[idx] = prob[idx]
            utils.save_volume(label_list[labels], aff, h, path_result)


def compute_hard_volumes(labels_dir,
                         voxel_volume=None,
                         path_label_list=None,
                         skip_background=True,
                         path_numpy_result=None,
                         path_csv_result=None,
                         FS_sort=False):
    """Compute hard volumes of structures for all label maps in a folder.
    :param labels_dir: path of directory with input label maps
    :param voxel_volume: (optional) volume of the voxels. If None, it will be directly inferred from the file header.
    Set to 1 for a voxel count.
    :param path_label_list: (optional) list of labels to compute volumes for.
    Can be an int, a sequence, or a numpy array. If None, the volumes of all label values are computed for each subject.
    :param skip_background: (optional) whether to skip computing the volume of the background.
    If label_list is None, this assumes background value is 0.
    If label_list is not None, this assumes the background is the first value in label list.
    :param path_numpy_result: (optional) path where to write the result volumes as a numpy array.
    :param path_csv_result: (optional) path where to write the results as csv file.
    :param FS_sort: (optional) whether to sort the labels in FreeSurfer order.
    :return: numpy array with the volume of each structure for all subjects.
    Rows represent label values, and columns represent subjects.
    """

    # create result directories
    if path_numpy_result is not None:
        if not os.path.isdir(os.path.dirname(path_numpy_result)):
            os.mkdir(os.path.dirname(path_numpy_result))
    if path_csv_result is not None:
        if not os.path.isdir(os.path.dirname(path_csv_result)):
            os.mkdir(os.path.dirname(path_csv_result))

    # load or compute labels list
    label_list = utils.get_list_labels(path_label_list, labels_dir, FS_sort=FS_sort)

    # create csv volume file if necessary
    if path_csv_result is not None:
        if skip_background:
            cvs_header = [['subject'] + [str(lab) for lab in label_list[1:]]]
        else:
            cvs_header = [['subject'] + [str(lab) for lab in label_list]]
        with open(path_csv_result, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(cvs_header)
        csvFile.close()

    # loop over label maps
    path_labels = utils.list_images_in_folder(labels_dir)
    if skip_background:
        volumes = np.zeros((label_list.shape[0]-1, len(path_labels)))
    else:
        volumes = np.zeros((label_list.shape[0], len(path_labels)))
    for idx, path_label in enumerate(path_labels):
        utils.print_loop_info(idx, len(path_labels), 10)

        # load segmentation, and compute unique labels
        labels, _, _, _, _, _, subject_res = utils.get_volume_info(path_label, return_volume=True)
        if voxel_volume is None:
            voxel_volume = np.prod(subject_res)
        subject_volumes = edit_labels.compute_hard_volumes(labels, voxel_volume, label_list, skip_background)
        volumes[:, idx] = subject_volumes

        # compute volumes
        if path_csv_result is not None:
            subject_volumes = np.around(volumes[:, idx], 3)
            row = [os.path.basename(path_label)] + [str(vol) for vol in subject_volumes]
            with open(path_csv_result, 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()

    # write numpy array if necessary
    if path_numpy_result is not None:
        np.save(path_numpy_result, volumes)

    return volumes
