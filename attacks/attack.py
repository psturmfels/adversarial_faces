"""
This file contains the base class for all attack class modules.
"""

import tensorflow as tf
import numpy as np

class Attacker():
    def __init__(self, model):
        """
        Initializes the object.

        Args:
            model: A tf.keras.models.Model instance
        """
        self.model = model

    def self_distance_attack(self, image_batch, epsilon=0.1, **kwargs):
        """
        Attacks a batch of images by attempting to maximize the distance
        in embedding space between the modified image and the original image.

        Args:
            image_batch: A batch of images.
            epsilon: Maximum perturbation amount.
            kwargs: Varies depending on attack.
        """
        raise Exception('The attack function is not implemented' + \
                        ' for this class. Likely you have imported' + \
                        ' the wrong class.')

    def target_image_attack(self, image_batch, target_batch, epsilon=0.1, **kwargs):
        """
        Attacks a batch of images by attempting to minimize the distance in
        embedding space between the modified image and the target image.

        Args:
            image_batch: A batch of images. The images to perturb.
            target_batch: A batch of images. The target images.
            epsilon: Maximum perturbation amount
            kwargs: Varies depending on attack.
        """
        raise Exception('The attack function is not implemented' + \
                        ' for this class. Likely you have imported' + \
                        ' the wrong class.')

    def _zero_one_norm(self, x):
        """
        Normalizes a tensor to be in the range [0, 1].

        Args:
            x: A tensor
        """
        min_x, max_x = tf.reduce_min(x), tf.reduce_max(x)
        return (x - min_x) / (max_x - min_x)

    def _l2_normalize(self, x, axis=-1, epsilon=1e-10):
        """
        Normalizes an embedding to have unit length in the l2 metric.

        Args:
            x: A batch of tensorflow embeddings.
        """
        output = x / tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x),
                                                      axis=axis,
                                                      keepdims=True),
                                        epsilon))
        return output

    def _l2_norm(self, x, axis=1):
        """
        Returns the L2 norms of a batch of vectors.

        Args:
            x: A batch of tensors
        """
        if axis is not None:
            return tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis))
        else:
            return tf.sqrt(tf.reduce_sum(tf.square(x)))

    def _l2_distance(self, x, y):
        """
        Returns the L2 distance between two batches of vectors.

        Args:
            x, y: Two TF tensors that represent batches of vectors.
        """
        return self._l2_norm(x - y)

    def _generate_noise(self,
                        epsilon,
                        image_batch,
                        norm_type='inf'):
        """
        Generates a random vector approximately on the surface of the
        norm_type-ball of radius epsilon

        Args:
            epsilon: Scaling parameter
        """
        if norm_type == 'inf':
            r = tf.random.uniform(shape=image_batch.shape,
                                  minval=-epsilon,
                                  maxval=epsilon,
                                  dtype=image_batch.dtype)
        elif norm_type == 'l2':
            dim = tf.reduce_prod(image_batch.shape[1:])

            x = tf.random.normal(shape=image_batch.shape, dtype=image_batch.dtype)
            norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))
            w = tf.pow(tf.random.uniform(shape=(image_batch.shape[0]) + \
                                               (1,) * len(image_batch.shape[1:]),
                                         dtype=image_batch.dtype),
                       1.0 / tf.cast(dim, image_batch.dtype))

            r = epsilon * tf.reshape(w * x / norm, image_batch.shape)
        else:
            raise ValueError('Unrecognized norm `{}`'.format(norm_type))

        return r