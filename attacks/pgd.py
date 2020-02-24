"""
Implements the Fast Gradient Sign attack method.
"""

import tensorflow as tf
from .attack import Attacker
from .fgsm import FGSMAttacker

class PGDAttacker(Attacker):
    """
    Runs Projected Gradient Descent.
    """

    def self_distance_attack(self, image_batch, epsilon=0.1, **kwargs):
        """
        Attacks a batch of images using PGD using the self-distance
        strategy.

        Args:
            image_batch: A batch of images.
            epsilon: Maximum perturbation amount.
            kwargs: Varies depending on attack.
        """
        if 'bounds' not in kwargs:
            kwargs['bounds'] = [tf.reduce_min(image_batch), tf.reduce_max(image_batch)]
        if 'num_iters' not in kwargs:
            kwargs['num_iters'] = 40
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.03

        image_batch = tf.convert_to_tensor(image_batch)

        original_embedding = self.model(image_batch)
        original_embedding = self._l2_normalize(original_embedding)

        gaussian_noise = self._generate_noise(kwargs['alpha'], image_batch)
        perturbed_image_batch = image_batch + gaussian_noise

        for i in range(kwargs['num_iters']):
            with tf.GradientTape() as tape:
                tape.watch(perturbed_image_batch)
                noisy_embedding = self.model(perturbed_image_batch)
                noisy_embedding = self._l2_normalize(noisy_embedding)
                difference = tf.sqrt(tf.reduce_sum(tf.square(noisy_embedding - original_embedding)))

            gradient = tape.gradient(difference, perturbed_image_batch)
            sign_of_gradient = tf.cast(tf.sign(gradient), dtype=perturbed_image_batch.dtype)

            perturbed_image_batch = perturbed_image_batch + sign_of_gradient * kwargs['alpha']
            perturbed_image_batch = tf.clip_by_value(perturbed_image_batch, image_batch - epsilon, image_batch + epsilon)
            perturbed_image_batch = tf.clip_by_value(perturbed_image_batch, kwargs['bounds'][0], kwargs['bounds'][1])

        return perturbed_image_batch

    def target_image_attack(self, image_batch, target_batch, epsilon=0.1, **kwargs):
        """
        Attacks a batch of images using PGD using the
        target-image strategy.

        Args:
            image_batch: A batch of images. The images to perturb.
            target_batch: A batch of images. The target images.
            epsilon: Maximum perturbation amount
            kwargs: Varies depending on attack.
        """
        if 'bounds' not in kwargs:
            kwargs['bounds'] = [tf.reduce_min(image_batch), tf.reduce_max(image_batch)]
        if 'num_iters' not in kwargs:
            kwargs['num_iters'] = 40
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.03

        image_batch  = tf.convert_to_tensor(image_batch)
        target_batch = tf.convert_to_tensor(target_batch)

        target_embedding = self.model(target_batch)
        target_embedding = self._l2_normalize(target_embedding)

        perturbed_image_batch = image_batch

        for i in range(kwargs['num_iters']):
            with tf.GradientTape() as tape:
                tape.watch(perturbed_image_batch)
                batch_embedding = self.model(perturbed_image_batch)
                batch_embedding = self._l2_normalize(batch_embedding)
                difference = tf.sqrt(tf.reduce_sum(tf.square(target_embedding - batch_embedding)))
            gradient = tape.gradient(difference, perturbed_image_batch)
            sign_of_gradient = tf.cast(tf.sign(gradient), perturbed_image_batch.dtype)

            # Subtract the gradient because we want to minimize the distance
            perturbed_image_batch = perturbed_image_batch - sign_of_gradient * kwargs['alpha']
            perturbed_image_batch = tf.clip_by_value(perturbed_image_batch, image_batch - epsilon, image_batch + epsilon)
            perturbed_image_batch = tf.clip_by_value(perturbed_image_batch, kwargs['bounds'][0], kwargs['bounds'][1])

        return perturbed_image_batch

