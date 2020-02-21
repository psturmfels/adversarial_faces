import tensorflow as tf
import numpy as np
import os

def set_up_environment(mem_frac=None, visible_devices=None, min_log_level='3'):
    """
    A helper function to set up a tensorflow environment.

    Args:
        mem_frac: Fraction of memory to limit the gpu to. If set to None,
                  turns on memory growth instead.
        visible_devices: A string containing a comma-separated list of
                         integers designating the gpus to run on.
        min_log_level: One of 0, 1, 2, or 3.
    """
    if visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(min_log_level)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if mem_frac is not None:
                    memory_limit = int(10000 * mem_frac)
                    config = [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit)]
                    tf.config.experimental.set_virtual_device_configuration(gpu, config)
                else:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            print(error)

def prewhiten(x):
    """
    A helper function to whiten an image, or a batch of images.

    Args:
        x: An image or batch of images.
    """
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def maximum_center_crop(x):
    """
    A helper function to crop an image to the maximum center crop.

    Args:
        x: An image.
    """
    minimum_dimension = min(x.shape[0], x.shape[1])
    extension = int(minimum_dimension / 2)
    center = (int(x.shape[0] / 2), int(x.shape[1] / 2))

    x = x[center[0] - extension:center[0] + extension,
          center[1] - extension:center[1] + extension]
    return x

def l2_normalize(x, axis=-1, epsilon=1e-10):
    """
    Normalizes an embedding to have unit length in the l2 metric.

    Args:
        x: A batch of numpy embeddings
    """
    output = x / np.sqrt(np.maximum(np.sum(np.square(x),
                                           axis=axis,
                                           keepdims=True),
                                    epsilon))
    return output
