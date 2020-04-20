import tensorflow as tf
import keras.layers as KL


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
