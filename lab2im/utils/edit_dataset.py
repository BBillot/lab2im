# python imports
import os
import numpy as np

# project imports
from lab2im.utils import utils, edit_volume


def check_images_and_labels(image_dir, labels_dir):
    """Check if corresponding images and labels have the same affine matrices and shapes.
    Labels are matched to images by sorting order.
    :param image_dir: path of directory with input images
    :param labels_dir: path of directory with corresponding label maps
    """

    # list images and labels
    path_images = utils.list_images_in_folder(image_dir)
    path_labels = utils.list_images_in_folder(labels_dir)
    assert len(path_images) == len(path_labels), 'different number of files in image_dir and labels_dir'

    # loop over images and labels
    for idx, (path_image, path_label) in enumerate(zip(path_images, path_labels)):
        utils.print_loop_info(idx, len(image_dir), 10)

        # load images and labels
        im, aff_im, h_im = utils.load_volume(path_image, im_only=False)
        lab, aff_lab, h_lab = utils.load_volume(path_label, im_only=False)
        aff_im_list = np.round(aff_im, 2).tolist()
        aff_lab_list = np.round(aff_lab, 2).tolist()

        # check matching affine and shape
        if aff_lab_list != aff_im_list:
            print('aff mismatch :\n' + path_image)
            print(aff_im_list)
            print('\n' + path_label)
            print(aff_lab_list)
        if lab.shape != im.shape:
            print('shape mismatch :\n' + path_image)
            print(im.shape)
            print('\n' + path_label)
            print(lab.shape)


def crop_dataset_to_same_size(labels_dir,
                              result_dir,
                              image_dir=None,
                              image_result_dir=None,
                              margin=5,
                              cropping_indices=None,
                              recompute=True):
    """Crop all label maps in a directory to the minimum possible size, with a margin.
    This function assumes all the label maps have the same size.
    If images are provided, they are cropped like their corresponding label maps.
    :param labels_dir: path of directory with input label maps
    :param result_dir: path of directory where cropped label maps will be writen
    :param image_dir: (optional) if not None, the cropping will be applied to all images in this directory
    :param image_result_dir: (optional) path of directory where cropped images will be writen
    :param margin: (optional) margin to apply around the cropping indices
    :param cropping_indices: (optional) if not None, the whole dataset will be cropped around those indices
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    if image_dir is not None:
        assert image_result_dir is not None, 'image_result_dir should not be None if image_dir is specified'
        if not os.path.isdir(image_result_dir):
            os.mkdir(image_result_dir)

    # list labels and images
    path_labels = utils.list_images_in_folder(labels_dir)
    if image_dir is not None:
        path_images = utils.list_images_in_folder(image_dir)
    else:
        path_images = [None] * len(path_labels)
    labels_shape, _, n_dims, _, _, _ = utils.get_volume_info(path_labels[0])

    # get cropping indices if not specified
    if cropping_indices is None:
        min_cropping = np.array(labels_shape, dtype='int')
        max_cropping = np.zeros(n_dims, dtype='int')
        print('getting cropping indices')
        for idx, path_labels in enumerate(path_labels):
            utils.print_loop_info(idx, len(path_labels), 10)
            label = utils.load_volume(path_labels)
            _, cropping = edit_volume.crop_volume_around_region(label, margin=margin)
            min_cropping = np.minimum(min_cropping, cropping[:n_dims], dtype='int')
            max_cropping = np.maximum(max_cropping, cropping[n_dims:], dtype='int')
        cropping_indices = np.concatenate([min_cropping, max_cropping])

    # loop over label maps
    print('\ncropping images')
    for idx, (path_label, path_image) in enumerate(zip(path_labels, path_images)):
        utils.print_loop_info(idx, len(path_labels), 10)

        # crop label map
        path_result = os.path.join(result_dir, os.path.basename(path_label))
        if (not os.path.isfile(path_result)) | recompute:
            label, aff, h = utils.load_volume(path_label, im_only=False)
            label, aff = edit_volume.crop_volume_with_idx(label, cropping_indices, aff=aff)
            utils.save_volume(label, aff, h, path_result)

        # crop image if necessary
        if path_image is not None:
            path_result = os.path.join(image_result_dir, os.path.basename(path_image))
            if (not os.path.isfile(path_result)) | recompute:
                im, aff, h = utils.load_volume(path_image, im_only=False)
                im, aff = edit_volume.crop_volume_with_idx(im, cropping_indices, aff=aff)
                utils.save_volume(im, aff, h, path_result)


def subdivide_dataset_to_patches(patch_shape,
                                 image_dir=None,
                                 image_result_dir=None,
                                 labels_dir=None,
                                 labels_result_dir=None,
                                 full_background=True):
    """This function subdivides images and/or label maps into several smaller patches of specified shape.
    :param patch_shape: shape of patches to create. Can either be an int, a sequence, or a 1d numpy array.
    :param image_dir: (optional) path of directory with input images
    :param image_result_dir: (optional) path of directory where image patches will be writen
    :param labels_dir: (optional) path of directory with input label maps
    :param labels_result_dir: (optional) path of directory where label map patches will be writen
    :param full_background: (optional) whether to keep patches only labelled as background (only if label maps are
    provided).
    """

    # create result dir and list images and label maps
    assert (image_dir is not None) | (labels_dir is not None), \
        'at least one of image_dir or labels_dir should not be None.'
    if image_dir is not None:
        assert image_result_dir is not None, 'image_result_dir should not be None if image_dir is specified'
        if not os.path.isdir(image_result_dir):
            os.mkdir(image_result_dir)
        path_images = utils.list_images_in_folder(image_dir)
    else:
        path_images = None
    if labels_dir is not None:
        assert labels_result_dir is not None, 'labels_result_dir should not be None if labels_dir is specified'
        if not os.path.isdir(labels_result_dir):
            os.mkdir(labels_result_dir)
        path_labels = utils.list_images_in_folder(labels_dir)
    else:
        path_labels = None
    if path_images is None:
        path_images = [None] * len(path_labels)
    if path_labels is None:
        path_labels = [None] * len(path_images)

    # reformat path_shape
    patch_shape = utils.reformat_to_list(patch_shape)
    n_dims, _ = utils.get_dims(patch_shape)

    # loop over images and labels
    for idx, (path_image, path_label) in enumerate(zip(path_images, path_labels)):
        utils.print_loop_info(idx, len(path_images), 10)

        # load image and labels
        if path_image is not None:
            im, aff_im, h_im = utils.load_volume(path_image, im_only=False, squeeze=False)
        else:
            im = aff_im = h_im = None
        if path_label is not None:
            lab, aff_lab, h_lab = utils.load_volume(path_label, im_only=False, squeeze=True)
        else:
            lab = aff_lab = h_lab = None

        # get volume shape
        if path_image is not None:
            shape = im.shape
        else:
            shape = lab.shape

        # crop image and label map to size divisible by patch_shape
        new_size = np.array([utils.find_closest_number_divisible_by_m(shape[i], patch_shape[i]) for i in range(n_dims)])
        crop = np.round((np.array(shape) - new_size) / 2).astype('int')
        crop = np.concatenate((crop, crop + new_size), axis=0)
        if (im is not None) & (n_dims == 2):
            im = im[crop[0]:crop[2], crop[1]:crop[3], ...]
        elif (im is not None) & (n_dims == 3):
            im = im[crop[0]:crop[3], crop[1]:crop[4], crop[2]:crop[5], ...]
        if (lab is not None) & (n_dims == 2):
            lab = lab[crop[0]:crop[2], crop[1]:crop[3], ...]
        elif (lab is not None) & (n_dims == 3):
            lab = lab[crop[0]:crop[3], crop[1]:crop[4], crop[2]:crop[5], ...]

        # loop over patches
        n_im = 0
        n_crop = (new_size / patch_shape).astype('int')
        for i in range(n_crop[0]):
            i *= patch_shape[0]
            for j in range(n_crop[1]):
                j *= patch_shape[1]

                if n_dims == 2:

                    # crop volumes
                    if lab is not None:
                        temp_la = lab[i:i+patch_shape[0], j:j+patch_shape[1], ...]
                    else:
                        temp_la = None
                    if im is not None:
                        temp_im = im[i:i + patch_shape[0], j:j + patch_shape[1], ...]
                    else:
                        temp_im = None

                    # write patches
                    if temp_la is not None:
                        if full_background | (not (temp_la == 0).all()):
                            n_im += 1
                            utils.save_volume(temp_la, aff_lab, h_lab, os.path.join(labels_result_dir,
                                              path_label.replace('.nii.gz', '_%d.nii.gz' % n_im)))
                            if temp_im is not None:
                                utils.save_volume(temp_im, aff_im, h_im, os.path.join(image_result_dir,
                                                  path_label.replace('.nii.gz', '_%d.nii.gz' % n_im)))
                    else:
                        utils.save_volume(temp_im, aff_im, h_im, os.path.join(image_result_dir,
                                          path_label.replace('.nii.gz', '_%d.nii.gz' % n_im)))

                if n_dims == 3:
                    for k in range(n_crop[2]):
                        k *= patch_shape[2]

                        # crop volumes
                        if lab is not None:
                            temp_la = lab[i:i + patch_shape[0], j:j + patch_shape[1], k:k+patch_shape[2], ...]
                        else:
                            temp_la = None
                        if im is not None:
                            temp_im = im[i:i + patch_shape[0], j:j + patch_shape[1], k:k + patch_shape[2], ...]
                        else:
                            temp_im = None

                        # write patches
                        if temp_la is not None:
                            if full_background | (not (temp_la == 0).all()):
                                n_im += 1
                                utils.save_volume(temp_la, aff_lab, h_lab, os.path.join(labels_result_dir,
                                                  path_label.replace('.nii.gz', '_%d.nii.gz' % n_im)))
                                if temp_im is not None:
                                    utils.save_volume(temp_im, aff_im, h_im, os.path.join(image_result_dir,
                                                      path_label.replace('.nii.gz', '_%d.nii.gz' % n_im)))
                        else:
                            utils.save_volume(temp_im, aff_im, h_im, os.path.join(image_result_dir,
                                              path_label.replace('.nii.gz', '_%d.nii.gz' % n_im)))
