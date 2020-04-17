# python imports
import math
import numpy as np
import numpy.random as npr

# project imports
import utils


def build_model_input_generator(labels_paths,
                                generation_label_list,
                                n_neutral,
                                labels_shape,
                                bias_field_tens_shape,
                                n_channels,
                                classes_list=None,
                                means_range=None,
                                std_devs_range=None,
                                use_specific_stats_for_channel=False,
                                intensity_constraints='rl_grouping',
                                apply_affine_trans=True,
                                scaling_range=None,
                                rotation_range=None,
                                shearing_range=None,
                                apply_nonlin_trans=True,
                                nonlin_shape_fact=0.0625,
                                nonlin_std_dev=3,
                                apply_bias_field=True,
                                bias_shape_fact=0.025,
                                bias_field_std_dev=0.3,
                                blur_background=True,
                                background_paths=None,
                                head=True,
                                batch_size=1,
                                in_memory=False):

    # get label info
    n_lab = np.size(generation_label_list)
    n_sided = int((n_lab - n_neutral) / 2)

    # If necessary, load data into memory
    dataset = []
    if in_memory is True:
        print('Loading full dataset in memory')
        for i in range(len(labels_paths)):
            y = utils.load_volume(labels_paths[i], dtype='int')
            dataset.append(y)
        print('Loading complete!')

    # Generate!
    while True:

        # randomly pick as many images as batch_size
        indices = npr.randint(len(labels_paths), size=batch_size)

        # initialise input tensors
        y_all = []
        means_all = []
        std_devs_all = []
        aff_all = []
        nonlinear_field_all = []
        bias_field_all = []

        for idx in indices:

            # add labels to inputs
            if in_memory is True:
                y = np.squeeze(dataset[idx])
            else:
                y = utils.load_volume(labels_paths[idx], dtype='int')
                if background_paths is not None:
                    idx_258 = np.where(y == 258)
                    if np.any(idx_258):
                        background_idx = npr.randint(len(background_paths))
                        background = utils.load_volume(background_paths[background_idx], dtype='int')
                        background_shape = background.shape
                        if np.all(np.array(background_shape) == background_shape[0]):  # flip if same dimensions
                            background = np.flip(background, tuple([i for i in range(3) if np.random.normal() > 0]))
                        assert background.shape == y.shape, \
                            'background patches should have same shape than training labels. ' \
                            'Had {0} and {1}'.format(background.shape, y.shape)
                        y[idx_258] = background[idx_258]
            y_all.append(utils.add_axis(y, axis=-2))

            # add means and standard deviations to inputs
            means = np.empty((n_lab, 0))
            std_devs = np.empty((n_lab, 0))
            for channel in range(n_channels):
                # retrieve channel specific stats if necessary
                if use_specific_stats_for_channel:
                    tmp_means_range = means_range[2*channel:2*channel + 2, :]
                    tmp_std_devs_range = std_devs_range[2*channel:2*channel + 2, :]
                else:
                    tmp_means_range = means_range
                    tmp_std_devs_range = std_devs_range
                # draw means and std devs from priors
                if intensity_constraints == 'no_rules':
                    tmp_means, tmp_stds = means_stds_no_rules(n_lab,
                                                              tmp_means_range,
                                                              tmp_std_devs_range)
                elif intensity_constraints == 'rl_grouping':
                    tmp_means, tmp_stds = means_stds_with_rl_grouping(n_sided,
                                                                      n_neutral,
                                                                      tmp_means_range,
                                                                      tmp_std_devs_range)
                elif intensity_constraints == 'classes':
                    tmp_means, tmp_stds = means_stds_with_classes(classes_list,
                                                                  tmp_means_range,
                                                                  tmp_std_devs_range)
                elif intensity_constraints == 'classes_with_relations':
                    tmp_means, tmp_stds = means_stds_fs_labels_with_relations(tmp_means_range,
                                                                              tmp_std_devs_range,
                                                                              head=head)
                elif intensity_constraints == 'stats':
                    tmp_means, tmp_stds = means_stds_with_stats(n_sided,
                                                                n_neutral,
                                                                tmp_means_range,
                                                                tmp_std_devs_range)
                elif intensity_constraints == 'classes_with_stats':
                    tmp_means, tmp_stds = means_stds_classes_with_stats(classes_list,
                                                                        tmp_means_range,
                                                                        tmp_std_devs_range)
                else:
                    raise ValueError('intensity_mode should be in [no_rules, rl_grouping, classes, '
                                     'classes_with_relations, stats], got %s' % intensity_constraints)
                if blur_background:
                    tmp_means[0] = np.random.uniform(low=0, high=150)
                    tmp_stds[0] = np.random.uniform(low=0, high=15)
                else:
                    tmp_means[0] = 0
                    tmp_stds[0] = 0
                means = np.concatenate([means, tmp_means], axis=1)
                std_devs = np.concatenate([std_devs, tmp_stds], axis=1)
            means_all.append(utils.add_axis(means))
            std_devs_all.append(utils.add_axis(std_devs))

            # add inputs according to augmentation specification
            if apply_affine_trans:
                n_dims, _ = utils.get_dims(labels_shape)
                # get affine transformation: rotate, scale, shear (translation done during random cropping)
                scaling = utils.draw_value_from_distribution(scaling_range, size=n_dims, centre=1, default_range=.15)
                if n_dims == 2:
                    rotation_angle = utils.draw_value_from_distribution(rotation_range, default_range=15.0)
                else:
                    rotation_angle = utils.draw_value_from_distribution(rotation_range, size=n_dims, default_range=15.0)
                shearing = utils.draw_value_from_distribution(shearing_range, size=n_dims**2-n_dims, default_range=.01)
                aff = create_affine_transformation_matrix(n_dims, scaling, rotation_angle, shearing)
                aff_all.append(utils.add_axis(aff))

            if apply_nonlin_trans:
                deform_shape = utils.get_resample_shape(labels_shape, nonlin_shape_fact, len(labels_shape))
                nonlinear_field = npr.normal(loc=0, scale=nonlin_std_dev * npr.rand(), size=deform_shape)
                nonlinear_field_all.append(utils.add_axis(nonlinear_field))

            if apply_bias_field:
                bias_shape = utils.get_resample_shape(bias_field_tens_shape[:-1], bias_shape_fact, n_channels=1)
                bias_field = npr.normal(loc=0, scale=bias_field_std_dev * npr.rand(), size=bias_shape)
                bias_field_all.append(utils.add_axis(bias_field))

        # build list of inputs to augmentation model
        inputs_vals = [y_all, means_all, std_devs_all]
        if apply_affine_trans:
            inputs_vals.append(aff_all)
        if apply_nonlin_trans:
            inputs_vals.append(nonlinear_field_all)
        if apply_bias_field:
            inputs_vals.append(bias_field_all)

        # put images and labels (concatenated if batch_size>1) into a tuple of 2 elements: (cat_images, cat_labels)
        if batch_size > 1:
            inputs_vals = [np.concatenate(item, 0) for item in inputs_vals]
        else:
            inputs_vals = [item[0] for item in inputs_vals]

        yield inputs_vals


def means_stds_no_rules(n_lab, means_range, std_devs_range):

    # draw values
    means = utils.add_axis(utils.draw_value_from_distribution(means_range, n_lab, 'uniform', 125., 100.), -1)
    stds = utils.add_axis(utils.draw_value_from_distribution(std_devs_range, n_lab, 'uniform', 15., 10.), -1)

    return means, stds


def means_stds_with_rl_grouping(n_sided, n_neutral, means_range, std_devs_range):

    # draw values
    n_samples = n_sided + n_neutral
    means = utils.add_axis(utils.draw_value_from_distribution(means_range, n_samples, 'uniform', 125., 100.), -1)
    stds = utils.add_axis(utils.draw_value_from_distribution(std_devs_range, n_samples, 'uniform', 15., 10.), -1)

    # regroup neutral and sided labels
    means = np.concatenate([means[:n_neutral], means[n_neutral:], means[n_neutral:]])
    stds = np.concatenate([stds[:n_neutral], stds[n_neutral:], stds[n_neutral:]])

    return means, stds


def means_stds_with_classes(classes_list, means_range, std_devs_range):

    # get unique list of classes and reorder them from 0 to N-1
    _, idx = np.unique(classes_list, return_index=True)
    unique_classes = np.sort(classes_list[np.sort(idx)])
    n_stats = len(unique_classes)

    # reformat classes_list
    _, classes_lut = utils.rearrange_label_list(classes_list)
    classes_list = (classes_lut[classes_list]).astype('int')

    # draw values
    means = utils.add_axis(utils.draw_value_from_distribution(means_range, n_stats, 'uniform', 125., 100.), -1)
    stds = utils.add_axis(utils.draw_value_from_distribution(std_devs_range, n_stats, 'uniform', 15., 10.), -1)

    # reorder values
    means = means[classes_list]
    stds = stds[classes_list]

    return means, stds


def means_stds_fs_labels_with_relations(means_range, std_devs_range, min_diff=15, head=True):

    # draw gm wm and csf means
    gm_wm_csf_means = np.zeros(3)
    while (abs(gm_wm_csf_means[1] - gm_wm_csf_means[0]) < min_diff) | \
          (abs(gm_wm_csf_means[1] - gm_wm_csf_means[2]) < min_diff) | \
          (abs(gm_wm_csf_means[0] - gm_wm_csf_means[2]) < min_diff):
        gm_wm_csf_means = utils.add_axis(utils.draw_value_from_distribution(means_range, 3, 'uniform', 125., 100.), -1)

    # apply relations
    wm = gm_wm_csf_means[0]
    gm = gm_wm_csf_means[1]
    csf = gm_wm_csf_means[2]
    csf_like = csf * npr.uniform(low=0.95, high=1.05)
    alpha_thalamus = npr.uniform(low=0.4, high=0.9)
    thalamus = alpha_thalamus*gm + (1-alpha_thalamus)*wm
    cerebellum_wm = wm * npr.uniform(low=0.7, high=1.3)
    cerebellum_gm = gm * npr.uniform(low=0.7, high=1.3)
    caudate = gm * npr.uniform(low=0.9, high=1.1)
    putamen = gm * npr.uniform(low=0.9, high=1.1)
    hippocampus = gm * npr.uniform(low=0.9, high=1.1)
    amygdala = gm * npr.uniform(low=0.9, high=1.1)
    accumbens = caudate * npr.uniform(low=0.9, high=1.1)
    pallidum = wm * npr.uniform(low=0.8, high=1.2)
    brainstem = wm * npr.uniform(low=0.8, high=1.2)
    alpha_ventralDC = npr.uniform(low=0.1, high=0.6)
    ventralDC = alpha_ventralDC*gm + (1-alpha_ventralDC)*wm
    alpha_choroid = npr.uniform(low=0.0, high=1.0)
    choroid = alpha_choroid*csf + (1-alpha_choroid)*wm

    # regroup structures
    neutral_means = [np.zeros(1), csf_like, csf_like, brainstem, csf]
    sided_means = [wm, gm, csf_like, csf_like, cerebellum_wm, cerebellum_gm, thalamus, caudate, putamen, pallidum,
                   hippocampus, amygdala, accumbens, ventralDC, choroid]

    # draw std deviations
    std = utils.add_axis(utils.draw_value_from_distribution(std_devs_range, 17, 'uniform', 15., 10.), -1)
    neutral_stds = [np.zeros(1), std[1], std[1], std[2], std[3]]
    sided_stds = [std[4], std[5], std[1], std[1], std[6], std[7], std[8], std[9], std[10], std[11], std[12], std[13],
                  std[14], std[15], std[16]]

    # add means and variances for extra head labels if necessary
    if head:
        # means
        extra_means = utils.add_axis(utils.draw_value_from_distribution(means_range, 2, 'uniform', 125., 100.), -1)
        skull = extra_means[0]
        soft_non_brain = extra_means[1]
        eye = csf * npr.uniform(low=0.95, high=1.05)
        optic_chiasm = wm * npr.uniform(low=0.8, high=1.2)
        vessel = csf * npr.uniform(low=0.7, high=1.3)
        neutral_means += [csf_like, optic_chiasm, skull, soft_non_brain, eye]
        sided_means.insert(-1, vessel)
        # std dev
        extra_std = utils.add_axis(utils.draw_value_from_distribution(std_devs_range, 4, 'uniform', 15., 10.), -1)
        neutral_stds += [std[1], extra_std[0], extra_std[1], extra_std[2], std[1]]
        sided_stds.insert(-1, extra_std[3])

    means = np.concatenate([np.array(neutral_means), np.array(sided_means), np.array(sided_means)])
    stds = np.concatenate([np.array(neutral_stds), np.array(sided_stds), np.array(sided_stds)])

    return means, stds


def means_stds_with_stats(n_sided, n_neutral, means_range, std_devs_range):

    # draw values
    n_samples = n_sided + n_neutral
    means = utils.add_axis(utils.draw_value_from_distribution(means_range[:, :n_samples], n_samples,
                                                              'normal', 125., 100.), -1)
    stds = utils.add_axis(utils.draw_value_from_distribution(std_devs_range[:, :n_samples], n_samples,
                                                             'normal', 15., 10.), -1)

    # regroup neutral and sided labels
    means = np.concatenate([means[:n_neutral], means[n_neutral:], means[n_neutral:]])
    stds = np.concatenate([stds[:n_neutral], stds[n_neutral:], stds[n_neutral:]])

    return means, stds


def means_stds_classes_with_stats(classes_list, means_range, std_devs_range):

    # get unique classes and corresponding stats
    unique_classes, unique_idx = np.unique(classes_list, return_index=True)
    n_unique = unique_classes.shape[0]
    unique_means_range = means_range[:, unique_idx]
    unique_std_devs_range = std_devs_range[:, unique_idx]

    # draw values
    unique_means = utils.add_axis(utils.draw_value_from_distribution(unique_means_range, n_unique,
                                                                     'normal', 125., 100.), -1)
    unique_stds = utils.add_axis(utils.draw_value_from_distribution(unique_std_devs_range, n_unique,
                                                                    'normal', 15., 10.), -1)

    # put stats back in order
    n_classes = classes_list.shape[0]
    means = np.zeros((n_classes, 1))
    stds = np.zeros((n_classes, 1))
    for idx_class, tmp_class in enumerate(unique_classes):
        means[classes_list == tmp_class] = unique_means[idx_class]
        stds[classes_list == tmp_class] = unique_stds[idx_class]

    return means, stds


def create_affine_transformation_matrix(n_dims, scaling=None, rotation=None, shearing=None, translation=None):
    """Create a 4x4 affine transformation matrix from specified values
    :param n_dims: integer
    :param scaling: list of 3 scaling values
    :param rotation: list of 3 angles (degrees) for rotations around 1st, 2nd, 3rd axis
    :param shearing: list of 6 shearing values
    :param translation: list of 3 values
    :return: 4x4 numpy matrix
    """

    T_scaling = np.eye(n_dims + 1)
    T_shearing = np.eye(n_dims + 1)
    T_translation = np.eye(n_dims + 1)

    if scaling is not None:
        T_scaling[np.arange(n_dims + 1), np.arange(n_dims + 1)] = np.append(scaling, 1)

    if shearing is not None:
        shearing_index = np.ones((n_dims + 1, n_dims + 1), dtype='bool')
        shearing_index[np.eye(n_dims + 1, dtype='bool')] = False
        shearing_index[-1, :] = np.zeros((n_dims + 1))
        shearing_index[:, -1] = np.zeros((n_dims + 1))
        T_shearing[shearing_index] = shearing

    if translation is not None:
        T_translation[np.arange(n_dims), n_dims * np.ones(n_dims, dtype='int')] = translation

    if n_dims == 2:
        if rotation is None:
            rotation = np.zeros(1)
        else:
            rotation = np.asarray(rotation) * (math.pi / 180)
        T_rot = np.eye(n_dims + 1)
        T_rot[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [np.cos(rotation[0]),
                                                                 np.sin(rotation[0]),
                                                                 np.sin(rotation[0]) * -1,
                                                                 np.cos(rotation[0])]
        return T_translation @ T_rot @ T_shearing @ T_scaling

    else:

        if rotation is None:
            rotation = np.zeros(n_dims)
        else:
            rotation = np.asarray(rotation) * (math.pi / 180)
        T_rot1 = np.eye(n_dims + 1)
        T_rot1[np.array([1, 2, 1, 2]), np.array([1, 1, 2, 2])] = [np.cos(rotation[0]),
                                                                  np.sin(rotation[0]),
                                                                  np.sin(rotation[0]) * -1,
                                                                  np.cos(rotation[0])]
        T_rot2 = np.eye(n_dims + 1)
        T_rot2[np.array([0, 2, 0, 2]), np.array([0, 0, 2, 2])] = [np.cos(rotation[1]),
                                                                  np.sin(rotation[1]) * -1,
                                                                  np.sin(rotation[1]),
                                                                  np.cos(rotation[1])]
        T_rot3 = np.eye(n_dims + 1)
        T_rot3[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [np.cos(rotation[2]),
                                                                  np.sin(rotation[2]),
                                                                  np.sin(rotation[2]) * -1,
                                                                  np.cos(rotation[2])]
        return T_translation @ T_rot3 @ T_rot2 @ T_rot1 @ T_shearing @ T_scaling
