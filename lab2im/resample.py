import ext.neuron.layers as nrn_layers


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
