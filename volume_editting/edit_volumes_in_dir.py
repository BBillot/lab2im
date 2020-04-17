# python imports
import os
import shutil
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
from keras.models import Model

# project imports
import utils
import building_blocks
from volume_editting import edit_volume


def mask_images_in_dir(image_dir, result_dir, mask_dir=None, threshold=0.1, dilate=0, erode=0, masking_value=0,
                       write_mask=False, mask_result_dir=None, recompute=True):
    """Mask all volumes in a folder, either with masks in a specified folder, or by keeping only the intensity values
    above a specified threshold.
    :param image_dir: path of directory with images to mask
    :param result_dir: path of directory where masked images will be writen
    :param mask_dir: (optional) path of directory containing masks. Masks are matched to images by sorting order.
    Mask volumes don't have to be boolean or 0/1 arrays as all strictly positive values are used to build the masks.
    Masks should have the same size as images. If images are multi-channel, masks can either be uni- or multi-channel.
    In the first case, the same mask is applied to all channels.
    :param threshold: (optional) If mask is None, masking is performed by keeping thresholding the input.
    :param dilate: (optional) number of voxels by which to dilate the provided or computed masks.
    :param erode: (optional) number of voxels by which to erode the provided or computed masks.
    :param masking_value: (optional) masking value
    :param write_mask: (optional) whether to write the applied masks
    :param mask_result_dir: (optional) path of resulting masks, if write_mask is True
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    if mask_result_dir is not None:
        if not os.path.isdir(mask_result_dir):
            os.mkdir(mask_result_dir)

    # loop over images
    path_images = utils.list_images_in_folder(image_dir)
    if mask_dir is not None:
        path_masks = utils.list_images_in_folder(mask_dir)
    else:
        path_masks = [None] * len(path_images)
    for idx, (path_image, path_mask) in enumerate(zip(path_images, path_masks)):
        utils.print_loop_info(idx, len(path_images), 10)

        # mask images
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            im, aff, h = utils.load_volume(path_image, im_only=False)
            if path_mask is not None:
                mask = utils.load_volume(path_mask)
            else:
                mask = None
            im = edit_volume.mask_volume(im, mask, threshold, dilate, erode, masking_value, write_mask)

            # write mask if necessary
            if write_mask:
                assert mask_result_dir is not None, 'if write_mask is True, mask_result_dir has to be specified as well'
                mask_result_path = os.path.join(mask_result_dir, os.path.basename(path_image))
                utils.save_volume(im[1], aff, h, mask_result_path)
                utils.save_volume(im[0], aff, h, path_result)
            else:
                utils.save_volume(im, aff, h, path_result)


def rescale_images_in_dir(image_dir, result_dir,
                          new_min=0, new_max=255,
                          min_percentile=0.025, max_percentile=0.975, use_positive_only=True,
                          recompute=True):
    """This function linearly rescales all volumes in image_dir between new_min and new_max.
    :param image_dir: path of directory with images to rescale
    :param result_dir: path of directory where rescaled images will be writen
    :param new_min: (optional) minimum value for the rescaled images.
    :param new_max: (optional) maximum value for the rescaled images.
    :param min_percentile: (optional) percentile for estimating robust minimum of each image.
    :param max_percentile: (optional) percentile for estimating robust maximum of each image.
    :param use_positive_only: (optional) whether to use only positive values when estimating the min and max percentile
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # loop over images
    path_images = utils.list_images_in_folder(image_dir)
    for idx, path_image in enumerate(path_images):
        utils.print_loop_info(idx, len(path_images), 10)

        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            im, aff, h = utils.load_volume(path_image, im_only=False)
            im = edit_volume.rescale_volume(im, new_min, new_max, min_percentile, max_percentile, use_positive_only)
            utils.save_volume(im, aff, h, path_result)


def crop_images_in_dir(image_dir, result_dir, cropping_margin=None, cropping_shape=None, recompute=True):
    """Crop all volumes in a folder by a given margin, or to a given shape.
    :param image_dir: path of directory with images to rescale
    :param result_dir: path of directory where cropped images will be writen
    :param cropping_margin: (optional) margin by which to crop the volume.
    Can be an int, a sequence or a 1d numpy array. Should be given if cropping_shape is None.
    :param cropping_shape: (optional) shape to which the volume will be cropped.
    Can be an int, a sequence or a 1d numpy array. Should be given if cropping_margin is None.
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # loop over images and masks
    path_images = utils.list_images_in_folder(image_dir)
    for idx, path_image in enumerate(path_images):
        utils.print_loop_info(idx, len(path_images), 10)

        # crop image
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            volume, aff, h = utils.load_volume(path_image, im_only=True)
            volume, aff = edit_volume.crop_volume(volume, cropping_margin, cropping_shape, aff)
            utils.save_volume(volume, aff, h, path_result)


def crop_images_around_region_in_dir(image_dir,
                                     result_dir,
                                     mask_dir=None,
                                     threshold=0.1,
                                     masking_labels=None,
                                     crop_margin=5,
                                     recompute=True):
    """Crop all volumes in a folder around a region, which is defined for each volume by a mask obtained by either
    1) directly providing it as input
    2) thresholding the input volume
    3) keeping a set of label values if the volume is a label map.
    :param image_dir: path of directory with images to crop
    :param result_dir: path of directory where cropped images will be writen
    :param mask_dir: (optional) path of directory of input masks
    :param threshold: (optional) lower bound to determine values to crop around
    :param masking_labels: (optional) if the volume is a label map, it can be cropped around a given set of labels by
    specifying them in masking_labels, which can either be a single int, a list or a 1d numpy array.
    :param crop_margin: (optional) cropping margin
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # list volumes and masks
    path_images = utils.list_images_in_folder(image_dir)
    if mask_dir is not None:
        path_masks = utils.list_images_in_folder(mask_dir)
    else:
        path_masks = [None] * len(path_images)

    # loop over images and masks
    for idx, (path_image, path_mask) in enumerate(zip(path_images, path_masks)):
        utils.print_loop_info(idx, len(path_images), 10)

        # crop image
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            volume, aff, h = utils.load_volume(path_image, im_only=True)
            if path_mask is not None:
                mask = utils.load_volume(path_mask)
            else:
                mask = None
            volume, cropping, aff = \
                edit_volume.crop_volume_around_region(volume, mask, threshold, masking_labels, crop_margin, aff)
            utils.save_volume(volume, aff, h, path_result)


def pad_images_in_dir(image_dir, result_dir, max_shape=None, padding_value=0, recompute=True):
    """Pads all the volumes in a folder to the same shape (either provided or computed).
    :param image_dir: path of directory with images to pad
    :param result_dir: path of directory where padded images will be writen
    :param max_shape: (optional) shape to pad the volumes to. Can be an int, a sequence or a 1d numpy array.
    If None, volumes will be padded to the shape of the biggest volume in image_dir.
    :param padding_value: (optional) value to pad the volumes with.
    :param recompute: (optional) whether to recompute result files even if they already exists
    :return: shape of the padded volumes.
    """

    # create result dir
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # list labels
    path_images = utils.list_images_in_folder(image_dir)

    # get maximum shape
    if max_shape is None:
        max_shape, aff, _, _, h, _ = utils.get_volume_info(path_images[0])
        for path_image in path_images[1:]:
            image_shape, aff, _, _, h, _ = utils.get_volume_info(path_image)
            max_shape = tuple(np.maximum(np.asarray(max_shape), np.asarray(image_shape)))
        max_shape = np.array(max_shape)

    # loop over label maps
    for idx, path_image in enumerate(path_images):
        utils.print_loop_info(idx, len(path_images), 5)

        # pad map
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            im, aff, h = utils.load_volume(path_image, im_only=True)
            im, aff = edit_volume.pad_volume(im, max_shape, padding_value, aff)
            utils.save_volume(im, aff, h, path_result)

    return max_shape


def flip_images_in_dir(image_dir, result_dir, axis=None, direction=None, recompute=True):
    """Flip all images in a diretory along a specified axis.
    If unknown, this axis can be replaced by an anatomical direction.
    :param image_dir: path of directory with images to flip
    :param result_dir: path of directory where flipped images will be writen
    :param axis: (optional) axis along which to flip the volume
    :param direction: (optional) if axis is None, the volume can be flipped along an anatomical direction:
    'rl' (right/left), 'ap' (anterior/posterior), 'si' (superior/inferior).
    :param recompute: (optional) whether to recompute result files even if they already exists
    """
    # create result dir
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # loop over images
    path_images = utils.list_images_in_folder(image_dir)
    for idx, path_image in enumerate(path_images):
        utils.print_loop_info(idx, len(path_images), 10)

        # flip image
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            im, aff, h = utils.load_volume(path_image, im_only=False)
            im = edit_volume.flip_volume(im, axis=axis, direction=direction, aff=aff)
            utils.save_volume(im, aff, h, path_result)


def align_images_in_dir(image_dir, result_dir, path_ref_image, recompute=True):
    """This function aligns all images in image_dir to the orientations of a reference image.
    :param image_dir: path of directory with images to align
    :param result_dir: path of directory where flipped images will be writen
    :param path_ref_image: path of a single reference image.
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # read ref axes and signs
    _, ref_aff, n_dims, _, _, _ = utils.get_volume_info(path_ref_image)
    ref_axes, ref_signs = edit_volume.get_ras_axes_and_signs(ref_aff, n_dims=n_dims)

    # loop over images
    path_images = utils.list_images_in_folder(image_dir)
    for idx, path_image in enumerate(path_images):
        utils.print_loop_info(idx, len(path_images), 10)

        # align image
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            im, aff, h = utils.load_volume(path_image, im_only=False)
            im_axes, im_signs = edit_volume.get_ras_axes_and_signs(aff, n_dims=n_dims)
            im = edit_volume.align_volume_to_ref(im, ref_axes, ref_signs, im_axes, im_signs)
            utils.save_volume(im, ref_aff, h, path_result)


def blur_images_in_dir(image_dir, result_dir, sigma, mask_dir=None, gpu=False, recompute=True):
    """This function blurs all the images in image_dir with kernels of the specified std deviations.
    :param image_dir: path of directory with images to blur
    :param result_dir: path of directory where blurred images will be writen
    :param sigma: standard deviation of the blurring gaussian kernels.
    Can be a number (isotropic blurring), or a sequence witht the same length as the number of dimensions of images.
    :param mask_dir: (optional) path of directory with masks of the region to blur.
    Images and masks are matched by sorting order.
    :param gpu: (optional) whether to use a fast gpu model for blurring
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # list images and masks
    path_images = utils.list_images_in_folder(image_dir)
    if mask_dir is not None:
        path_masks = utils.list_images_in_folder(mask_dir)
    else:
        path_masks = [None] * len(path_images)

    # loop over images
    previous_model_input_shape = None
    model = None
    for idx, (path_image, path_mask) in enumerate(zip(path_images, path_masks)):
        utils.print_loop_info(idx, len(path_images), 10)

        # load image
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            im, im_shape, aff, n_dims, _, h, image_res = utils.get_volume_info(path_image, return_volume=True)
            if path_mask is not None:
                mask = utils.load_volume(path_mask)
                assert mask.shape == im.shape, 'mask and image should have the same shape'
            else:
                mask = None

            # blur image
            if gpu:
                if (im_shape != previous_model_input_shape) | (model is None):
                    previous_model_input_shape = im_shape
                    image_in = [KL.Input(shape=im_shape + [1])]
                    sigma = utils.reformat_to_list(sigma, length=n_dims)
                    kernels_list = building_blocks.get_gaussian_1d_kernels(sigma)
                    image = building_blocks.blur_tensor(image_in[0], kernels_list, n_dims)
                    if mask is not None:
                        image_in.append(KL.Input(shape=im_shape + [1], dtype='float32'))  # mask
                        masked_mask = KL.Lambda(lambda x: tf.where(tf.greater(x, 0), tf.ones_like(x, dtype='float32'),
                                                                   tf.zeros_like(x, dtype='float32')))(image_in[1])
                        blurred_mask = building_blocks.blur_tensor(masked_mask, kernels_list, n_dims)
                        image = KL.Lambda(lambda x: x[0] / (x[1] + K.epsilon()))([image, blurred_mask])
                        image = KL.Lambda(lambda x: tf.where(tf.cast(x[1], dtype='bool'), x[0],
                                                             tf.zeros_like(x[0])))([image, masked_mask])
                    model = Model(inputs=image_in, outputs=image)
                if mask is None:
                    im = np.squeeze(model.predict(utils.add_axis(im, -2)))
                else:
                    im = np.squeeze(model.predict([utils.add_axis(im, -2), utils.add_axis(mask, -2)]))
            else:
                im = edit_volume.blur_volume(im, sigma, mask=mask)
            utils.save_volume(im, aff, h, path_result)


def create_mutlimodal_images(list_channel_dir, result_dir, recompute=True):
    """This function forms multimodal images by stacking channels located in different folders.
    :param list_channel_dir: list of all directories, each containing the same channel for allimages.
    Channels are matched between folders by sorting order.
    :param result_dir: path of directory where multimodal images will be writen
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    if not isinstance(list_channel_dir, list):
        raise TypeError('list_channel_dir should be a list')

    # gather path of all images for all channels
    list_channel_paths = [utils.list_images_in_folder(d) for d in list_channel_dir]
    n_images = len(list_channel_paths[0])
    n_channels = len(list_channel_dir)
    for channel_paths in list_channel_paths:
        if len(channel_paths) != n_images:
            raise ValueError('all directories should have the same number of files')

    # loop over images
    for idx in range(n_images):
        utils.print_loop_info(idx, n_images, 10)

        # stack all channels and save multichannel image
        path_result = os.path.join(result_dir, os.path.basename(list_channel_paths[0][idx]))
        if (not os.path.isfile(path_result)) | recompute:
            list_channels = list()
            tmp_aff = None
            tmp_h = None
            for channel_idx in range(n_channels):
                tmp_channel, tmp_aff, tmp_h = utils.load_volume(list_channel_paths[channel_idx][idx], im_only=False)
                list_channels.append(tmp_channel)
            im = np.stack(list_channels, axis=-1)
            utils.save_volume(im, tmp_aff, tmp_h, path_result)


def convert_images_in_dir_to_nifty(image_dir, result_dir, aff=None, recompute=True):
    """Converts all images in image_dir to nifty format.
    :param image_dir: path of directory with images to convert
    :param result_dir: path of directory where converted images will be writen
    :param aff: (optional) affine matrix in homogeneous coordinates with which to write the images.
    Can also be 'FS' to write images with FreeSurfer typical affine matrix.
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # loop over images
    path_images = utils.list_images_in_folder(image_dir)
    for idx, path_image in enumerate(path_images):
        utils.print_loop_info(idx, len(path_images), 10)

        # convert images to nifty format
        path_result = os.path.join(result_dir, os.path.basename(utils.strip_extension(path_image))) + '.nii.gz'
        if (not os.path.isfile(path_result)) | recompute:
            im, tmp_aff, h = utils.load_volume(path_image, im_only=False)
            if aff is not None:
                tmp_aff = aff
            utils.save_volume(im, tmp_aff, h, path_result)


def mri_convert_images_in_dir(image_dir,
                              result_dir,
                              interpolation=None,
                              reference_dir=None,
                              same_reference=False,
                              voxsize=None,
                              path_freesurfer='/usr/local/freesurfer',
                              mri_convert_path='/usr/local/freesurfer/bin/mri_convert.bin',
                              recompute=True):
    """This function launches mri_convert on all images contained in image_dir, and writes the results in result_dir.
    The interpolation type can be specified (i.e. 'nearest'), as well as a folder containing references for resampling.
    reference_dir can be the path of a single *image* if same_reference=True.
    :param image_dir: path of directory with images to convert
    :param result_dir: path of directory where converted images will be writen
    :param interpolation: (optional) interpolation type, can be 'inter' (default), 'cubic', 'nearest', 'trilinear'
    :param reference_dir: (optional) path of directory with reference images. References are matched to images by
    sorting order. If same_reference is false, references and images are matched by sorting order.
    :param same_reference: (optional) whether to use a single reference for all images.
    :param voxsize: (optional) resolution at which to resample converted image. Must be a list of length n_dims.
    :param path_freesurfer: (optional) path FreeSurfer home
    :param mri_convert_path: (optional) path mri_convert binary file
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # set up FreeSurfer
    os.environ['FREESURFER_HOME'] = path_freesurfer
    os.system(os.path.join(path_freesurfer, 'SetUpFreeSurfer.sh'))
    mri_convert = mri_convert_path + ' '

    # list images
    path_images = utils.list_images_in_folder(image_dir)
    if reference_dir is not None:
        if same_reference:
            path_references = [reference_dir] * len(path_images)
        else:
            path_references = utils.list_images_in_folder(reference_dir)
            assert len(path_references) == len(path_images), 'different number of files in image_dir and reference_dir'
    else:
        path_references = [None] * len(path_images)

    # loop over images
    for idx, (path_image, path_reference) in enumerate(zip(path_images, path_references)):
        utils.print_loop_info(idx, len(path_images), 10)

        # convert image
        path_result = os.path.join(result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_result)) | recompute:
            cmd = mri_convert + path_image + ' ' + path_result + ' -odt float'
            if interpolation is not None:
                cmd += ' -rt ' + interpolation
            if reference_dir is not None:
                cmd += ' -rl ' + path_reference
            if voxsize is not None:
                voxsize = utils.reformat_to_list(voxsize, dtype='float')
                cmd += ' --voxsize ' + ' '.join([str(np.around(v, 3)) for v in voxsize])
            os.system(cmd)


def samseg_images_in_dir(image_dir,
                         result_dir,
                         atlas_dir=None,
                         threads=4,
                         path_freesurfer='/usr/local/freesurfer',
                         keep_segm_only=True,
                         recompute=True):
    """This function launches samseg for all images contained in image_dir and writes the results in result_dir.
    If keep_segm_only=True, the result segmentation is copied in result_dir and SAMSEG's intermediate result dir is
    deleted.
    :param image_dir: path of directory with input images
    :param result_dir: path of directory where processed images folders (if keep_segm_only is False),
    or samseg segmentation (if keep_segm_only is True) will be writen
    :param atlas_dir: (optional) path of samseg atlas directory. If None, use samseg default atlas.
    :param threads: (optional) number of threads to use
    :param path_freesurfer: (optional) path FreeSurfer home
    :param keep_segm_only: (optional) whether to keep samseg result folders, or only samseg segmentations.
    :param recompute: (optional) whether to recompute result files even if they already exists
    """

    # create result dir
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # set up FreeSurfer
    os.environ['FREESURFER_HOME'] = path_freesurfer
    os.system(os.path.join(path_freesurfer, 'SetUpFreeSurfer.sh'))
    path_samseg = os.path.join(path_freesurfer, 'bin', 'run_samseg')

    # loop over images
    path_images = utils.list_images_in_folder(image_dir)
    for idx, path_image in enumerate(path_images):
        utils.print_loop_info(idx, len(path_images), 10)

        # build path_result
        path_im_result_dir = os.path.join(result_dir, utils.strip_extension(os.path.basename(path_image)))
        path_samseg_result = os.path.join(path_im_result_dir, ''.join(
            os.path.basename(path_image).split('.')[:-1]) + '_crispSegmentation.nii')
        if keep_segm_only:
            path_result = os.path.join(result_dir, os.path.basename(path_image))
        else:
            path_result = path_samseg_result

        # run samseg
        if (not os.path.isfile(path_result)) | recompute:
            cmd = path_samseg + ' -i ' + path_image + ' -o ' + path_im_result_dir + ' --threads ' + str(threads)
            if atlas_dir is not None:
                cmd += ' --a ' + atlas_dir
            os.system(cmd)

        # move segmentation to result_dir if necessary
        if keep_segm_only:
            if os.path.isfile(path_samseg_result):
                shutil.move(path_samseg_result, path_result)
            if os.path.isdir(path_im_result_dir):
                shutil.rmtree(path_im_result_dir)


def simulate_upsampled_anisotropic_images(image_dir,
                                          downsample_image_result_dir,
                                          resample_image_result_dir,
                                          data_res,
                                          labels_dir=None,
                                          downsample_labels_result_dir=None,
                                          slice_thickness=None,
                                          path_freesurfer='/usr/local/freesurfer/',
                                          gpu=False,
                                          recompute=True):
    """This function takes as input a set of HR images and creates two datasets with it:
    1) a set of LR images obtained by downsampling the HR images with nearest neighbour interpolation,
    2) a set of HR images obtained by resampling the LR images to native HR with linear interpolation.
    Additionally, this function can also create a set of LR labels from label maps corresponding to the input images.
    :param image_dir: path of directory with input images (only uni-model images supported)
    :param downsample_image_result_dir: path of directory where downsampled images will be writen
    :param resample_image_result_dir: path of directory where resampled images will be writen
    :param data_res: resolution of LR images. Can either be: an int, a float, a list or a numpy array.
    :param labels_dir: (optional) path of directory with label maps corresponding to input images
    :param downsample_labels_result_dir: (optional) path of directory where downsampled label maps will be writen
    :param slice_thickness: (optional) thickness of slices to simulate. Can be a number, a list or a numpy array.
    :param path_freesurfer: (optional) path freesurfer home, as this function uses mri_convert
    :param gpu: (optional) whether to use a fast gpu model for blurring
    :param recompute: (optional) whether to recompute result files even if they already exists
    """
    # create result dir
    if not os.path.isdir(resample_image_result_dir):
        os.mkdir(resample_image_result_dir)
    if not os.path.isdir(downsample_image_result_dir):
        os.mkdir(downsample_image_result_dir)
    if labels_dir is not None:
        assert downsample_labels_result_dir is not None, \
            'downsample_labels_result_dir should not be None if labels_dir is specified'
        if not os.path.isdir(downsample_labels_result_dir):
            os.mkdir(downsample_labels_result_dir)

    # set up FreeSurfer
    os.environ['FREESURFER_HOME'] = path_freesurfer
    os.system(os.path.join(path_freesurfer, 'SetUpFreeSurfer.sh'))
    mri_convert = os.path.join(path_freesurfer, 'bin/mri_convert.bin') + ' '

    # list images and labels
    path_images = utils.list_images_in_folder(image_dir)
    if labels_dir is not None:
        path_labels = utils.list_images_in_folder(labels_dir)
    else:
        path_labels = [None] * len(path_images)

    # initialisation
    _, _, n_dims, _, _, image_res = utils.get_volume_info(path_images[0], return_volume=False)
    data_res = np.squeeze(utils.reformat_to_n_channels_array(data_res, n_dims, n_channels=1))
    slice_thickness = utils.reformat_to_list(slice_thickness, length=n_dims)

    # loop over images
    previous_model_input_shape = None
    model = None
    for idx, (path_image, path_labels) in enumerate(zip(path_images, path_labels)):
        utils.print_loop_info(idx, len(path_images), 10)

        # downsample image
        path_im_downsampled = os.path.join(downsample_image_result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_im_downsampled)) | recompute:
            im, im_shape, aff, n_dims, _, h, image_res = utils.get_volume_info(path_image, return_volume=True)
            sigma = utils.get_std_blurring_mask_for_downsampling(data_res, image_res, thickness=slice_thickness)

            # blur image
            if gpu:
                if (im_shape != previous_model_input_shape) | (model is None):
                    previous_model_input_shape = im_shape
                    image_in = KL.Input(shape=im_shape + [1])
                    kernels_list = building_blocks.get_gaussian_1d_kernels(sigma)
                    kernels_list = [None if data_res[i] == image_res[i] else kernels_list[i] for i in range(n_dims)]
                    image = building_blocks.blur_tensor(image_in, kernels_list, n_dims)
                    model = Model(inputs=image_in, outputs=image)
                im = np.squeeze(model.predict(utils.add_axis(im, -2)))
            else:
                im = edit_volume.blur_volume(im, sigma, mask=None)
            utils.save_volume(im, aff, h, path_im_downsampled)

            # downsample blurred image
            voxsize = ' '.join([str(r) for r in data_res])
            cmd = mri_convert + path_im_downsampled + ' ' + path_im_downsampled + ' --voxsize ' + voxsize
            cmd += ' -odt float -rt nearest'
            os.system(cmd)

        # downsample labels if necessary
        if path_labels is not None:
            path_lab_downsampled = os.path.join(downsample_labels_result_dir, os.path.basename(path_labels))
            if (not os.path.isfile(path_lab_downsampled)) | recompute:
                voxsize = ' '.join([str(r) for r in data_res])
                cmd = mri_convert + path_labels + ' ' + path_lab_downsampled + ' --voxsize ' + voxsize
                cmd += ' -odt float -rt nearest'
                os.system(cmd)

        # upsample image
        path_im_upsampled = os.path.join(resample_image_result_dir, os.path.basename(path_image))
        if (not os.path.isfile(path_im_upsampled)) | recompute:
            cmd = mri_convert + path_im_downsampled + ' ' + path_im_upsampled + ' -rl ' + path_image + ' -odt float'
            os.system(cmd)


def check_images_in_dir(image_dir, check_unique_values=False):
    """Check if all volumes within the same folder share the same characteristics: shape, affine matrix, resolution.
    Also have option to check if all volumes have the same intensity values (useful for label maps).
    :return four lists, each containing the different values detected for a specific parameter among those to check."""

    # define information to check
    list_shape = list()
    list_aff = list()
    list_res = list()
    if check_unique_values:
        list_unique_values = list()
    else:
        list_unique_values = None

    # loop through files
    path_images = utils.list_images_in_folder(image_dir)
    for idx, path_image in enumerate(path_images):
        utils.print_loop_info(idx, len(path_images), 10)

        # get info
        im, im_shape, aff, _, _, h, data_res = utils.get_volume_info(path_image, return_volume=True)
        aff = np.round(aff[:3, :3], 2).tolist()
        data_res = np.round(np.array(data_res), 2).tolist()

        # add values to list if not already there
        if im_shape not in list_shape:
            list_shape.append(im_shape)
        if aff not in list_aff:
            list_aff.append(aff)
        if data_res not in list_res:
            list_res.append(data_res)
        if list_unique_values is not None:
            uni = np.unique(im).tolist()
            if uni not in list_unique_values:
                list_unique_values.append(uni)

    return list_shape, list_aff, list_res, list_unique_values
