"""
Implements the Fast Gradient Sign attack method.
"""

import tensorflow as tf
from .attack import Attacker

class FGSMAttacker(Attacker):
    """
    Runs the Fast Gradient Sign Method.
    """
    def self_distance_attack(self, image_batch, epsilon=0.1, **kwargs):
        """
        Attacks a batch of images using FGSM using the
        self-distance strategy.

        Args:
            image_batch: A batch of images.
            epsilon: Maximum perturbation amount.
            kwargs: Varies depending on attack.
        """
        image_batch = tf.convert_to_tensor(image_batch)

        original_embedding = self.model(image_batch)
        original_embedding = self._l2_normalize(original_embedding)

        gaussian_noise = self._generate_noise(epsilon, image_batch)
        noisy_image_batch = image_batch + gaussian_noise

        with tf.GradientTape() as tape:
            tape.watch(noisy_image_batch)
            noisy_embedding = self.model(noisy_image_batch)
            noisy_embedding = self._l2_normalize(noisy_embedding)
            difference = self._l2_distance(noisy_embedding, original_embedding)

        gradient = tape.gradient(difference, noisy_image_batch)
        sign_of_gradient = tf.cast(tf.sign(gradient), image_batch.dtype)

        perturbed_image_batch = noisy_image_batch + sign_of_gradient * epsilon
        perturbed_image_batch = tf.clip_by_value(perturbed_image_batch, image_batch - epsilon, image_batch + epsilon)
        return perturbed_image_batch

    def target_image_attack(self, image_batch, target_batch, epsilon=0.1, **kwargs):
        """
        Attacks a batch of images using FGSM using the
        target-image strategy.

        Args:
            image_batch: A batch of images. The images to perturb.
            target_batch: A batch of images. The target images.
            epsilon: Maximum perturbation amount
            kwargs: Varies depending on attack.
        """
        image_batch  = tf.convert_to_tensor(image_batch)
        target_batch = tf.convert_to_tensor(target_batch)

        target_embedding = self.model(target_batch)
        target_embedding = self._l2_normalize(target_embedding)

        with tf.GradientTape() as tape:
            tape.watch(image_batch)
            batch_embedding = self.model(image_batch)
            batch_embedding = self._l2_normalize(batch_embedding)
            difference = self._l2_distance(target_embedding, batch_embedding)

        gradient = tape.gradient(difference, image_batch)
        sign_of_gradient = tf.cast(tf.sign(gradient), image_batch.dtype)

        # Subtract the gradient because we want to minimize the distance
        perturbed_image_batch = image_batch - sign_of_gradient * epsilon
        return perturbed_image_batch
