"""
Implements the Projected Gradient Descent attack method.
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from .attack import Attacker

class PGDAttacker(Attacker):
    """
    Runs Projected Gradient Descent.
    """

    def _get_default_kwargs(self, kwargs, image_batch):
        """
        Gets some default values for existing hyper-parameters.
        """
        if 'bounds' not in kwargs:
            kwargs['bounds'] = [tf.reduce_min(image_batch), tf.reduce_max(image_batch)]
        if 'num_iters' not in kwargs:
            kwargs['num_iters'] = 400
        if 'patience' not in kwargs:
            kwargs['patience'] = 5
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.001
        if 'verbose' not in kwargs:
            kwargs['verbose'] = False
        return kwargs

    def self_distance_attack(self, image_batch, epsilon=0.1, **kwargs):
        """
        Attacks a batch of images using PGD using the self-distance
        strategy.

        Args:
            image_batch: A batch of images.
            epsilon: Maximum perturbation amount.
            kwargs: Varies depending on attack.
        """
        kwargs = self._get_default_kwargs(kwargs, image_batch)
        image_batch = tf.convert_to_tensor(image_batch)

        original_embedding = self.model(image_batch)
        original_embedding = self._l2_normalize(original_embedding)

        gaussian_noise = self._generate_noise(kwargs['alpha'], image_batch)
        perturbed_image_batch = image_batch + gaussian_noise

        previous_difference = 0.0
        best_perturbation = perturbed_image_batch
        patience_count = 0

        iterable = range(kwargs['num_iters'])
        if kwargs['verbose']:
            iterable = tqdm(iterable)

        for i in iterable:
            with tf.GradientTape() as tape:
                tape.watch(perturbed_image_batch)
                noisy_embedding = self.model(perturbed_image_batch)
                noisy_embedding = self._l2_normalize(noisy_embedding)
                difference = self._l2_distance(noisy_embedding, original_embedding)
                mean_difference = tf.reduce_mean(difference)

            if mean_difference > previous_difference:
                previous_difference = mean_difference
                best_perturbation = perturbed_image_batch
                patience_counnt = 0
            elif patience_count >= kwargs['patience']:
                break
            else:
                patience_count += 1

            gradient = tape.gradient(difference, perturbed_image_batch)
            sign_of_gradient = tf.cast(tf.sign(gradient), dtype=perturbed_image_batch.dtype)

            perturbed_image_batch = perturbed_image_batch + sign_of_gradient * kwargs['alpha']
            perturbed_image_batch = tf.clip_by_value(perturbed_image_batch, image_batch - epsilon, image_batch + epsilon)
            perturbed_image_batch = tf.clip_by_value(perturbed_image_batch, kwargs['bounds'][0], kwargs['bounds'][1])

        return best_perturbation

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
        kwargs = self._get_default_kwargs(kwargs, image_batch)
        image_batch  = tf.convert_to_tensor(image_batch)
        target_batch = tf.convert_to_tensor(target_batch)

        target_embedding = self.model(target_batch)
        target_embedding = self._l2_normalize(target_embedding)

        perturbed_image_batch = image_batch

        previous_difference = np.inf
        best_perturbation = perturbed_image_batch
        patience_count = 0

        iterable = range(kwargs['num_iters'])
        if kwargs['verbose']:
            iterable = tqdm(iterable)

        for i in iterable:
            with tf.GradientTape() as tape:
                tape.watch(perturbed_image_batch)
                batch_embedding = self.model(perturbed_image_batch)
                batch_embedding = self._l2_normalize(batch_embedding)
                difference = self._l2_distance(target_embedding, batch_embedding)
                mean_difference = tf.reduce_mean(difference)

            if mean_difference < previous_difference:
                previous_difference = mean_difference
                best_perturbation = perturbed_image_batch
                patience_counnt = 0
            elif patience_count >= kwargs['patience']:
                break
            else:
                patience_count += 1

            gradient = tape.gradient(difference, perturbed_image_batch)
            sign_of_gradient = tf.cast(tf.sign(gradient), perturbed_image_batch.dtype)

            # Subtract the gradient because we want to minimize the distance
            perturbed_image_batch = perturbed_image_batch - sign_of_gradient * kwargs['alpha']
            perturbed_image_batch = tf.clip_by_value(perturbed_image_batch, image_batch - epsilon, image_batch + epsilon)
            perturbed_image_batch = tf.clip_by_value(perturbed_image_batch, kwargs['bounds'][0], kwargs['bounds'][1])

        return best_perturbation

    def target_vector_attack(self, image_batch, target_embedding, normalize_target_embedding=True, epsilon=0.1, **kwargs):
        """
        Attacks a batch of images using PGD using the
        target-image strategy.

        Args:
            image_batch: A batch of images. The images to perturb.
            target_embedding: the target embeddings to send adversarial images to
            normalize_target_embedding: if True, l2 normalizes the target_embedding
            epsilon: Maximum perturbation amount
            kwargs: Varies depending on attack.
        """
        kwargs = self._get_default_kwargs(kwargs, image_batch)
        image_batch  = tf.convert_to_tensor(image_batch)
        target_embedding = tf.convert_to_tensor(target_embedding, dtype=tf.float32)
        if normalize_target_embedding:
            target_embedding = self._l2_normalize(target_embedding)
        perturbed_image_batch = image_batch

        previous_difference = np.inf
        best_perturbation = perturbed_image_batch
        patience_count = 0

        iterable = range(kwargs['num_iters'])
        if kwargs['verbose']:
            iterable = tqdm(iterable)

        for i in iterable:
            with tf.GradientTape() as tape:
                tape.watch(perturbed_image_batch)
                batch_embedding = self.model(perturbed_image_batch)
                batch_embedding = self._l2_normalize(batch_embedding)
                difference = self._l2_distance(target_embedding, batch_embedding)
                mean_difference = tf.reduce_mean(difference)

            if mean_difference < previous_difference:
                previous_difference = mean_difference
                best_perturbation = perturbed_image_batch
                patience_counnt = 0
            elif patience_count >= kwargs['patience']:
                break
            else:
                patience_count += 1

            gradient = tape.gradient(difference, perturbed_image_batch)
            sign_of_gradient = tf.cast(tf.sign(gradient), perturbed_image_batch.dtype)

            # Subtract the gradient because we want to minimize the distance
            perturbed_image_batch = perturbed_image_batch - sign_of_gradient * kwargs['alpha']
            perturbed_image_batch = tf.clip_by_value(perturbed_image_batch, image_batch - epsilon, image_batch + epsilon)
            perturbed_image_batch = tf.clip_by_value(perturbed_image_batch, kwargs['bounds'][0], kwargs['bounds'][1])

        return best_perturbation


class RobustPGDAttacker(Attacker):
    def _get_default_kwargs(self, kwargs, image_batch):
        """
        Gets some default values for existing hyper-parameters.
        """
        kwargs_for_this = {
                "num_iters": 2000,
                "epsilon": 0.5,
                "iters_no_rand": 10,
                "alpha": 0.001,
                "scratch_folder": "/home/ivan/adversarial_ivan_yoshi/epsilon_0.5_flip_crop_resize_bright_gaussian"
        }
        kwargs.update(kwargs_for_this)
        if 'bounds' not in kwargs:
            kwargs['bounds'] = [tf.reduce_min(image_batch), tf.reduce_max(image_batch)]
        if 'verbose' not in kwargs:
            kwargs['verbose'] = False
        return kwargs

    def _transform_batch(self, perturbed_image_batch):
        batch_input_to_model = tf.image.random_flip_left_right(perturbed_image_batch)
        batch_input_to_model = tf.image.random_brightness(batch_input_to_model, max_delta=0.25)
        batch_input_to_model = tf.image.random_crop(
            batch_input_to_model,
            [self.batch_size, self.orig_h - 10, self.orig_w - 10, 3]
        )
        batch_input_to_model = tf.image.resize(batch_input_to_model, [self.orig_h, self.orig_w])

        batch_input_to_model += tf.random.normal(batch_input_to_model.shape, 0.0, 0.5)
        return batch_input_to_model


    def target_vector_attack(
        self,
        image_batch,
        target_embedding,
        normalize_target_embedding=True,
        epsilon=0.1,
        **kwargs
    ):
        """
        Attacks a batch of images using PGD using the
        target-image strategy.
        Args:
            image_batch: A batch of images. The images to perturb.
            target_embedding: the target embeddings to send adversarial images to
            normalize_target_embedding: if True, l2 normalizes the target_embedding
            epsilon: Maximum perturbation amount
            kwargs: Varies depending on attack.
        """
        kwargs = self._get_default_kwargs(kwargs, image_batch)
        image_batch  = tf.convert_to_tensor(image_batch.copy())

        self.batch_size, self.orig_h, self.orig_w, self.orig_c = image_batch.shape.as_list()

        target_embedding = tf.convert_to_tensor(target_embedding, dtype=tf.float32)

        if normalize_target_embedding:
            target_embedding = self._l2_normalize(target_embedding)

        perturbed_image_batch = image_batch

        previous_difference = np.inf
        best_perturbation = perturbed_image_batch
        patience_count = 0

        iterable = range(kwargs['num_iters'])
        if kwargs['verbose']:
            iterable = tqdm(iterable)

        self.losses = []
        for i in iterable:
            with tf.GradientTape() as tape:
                tape.watch(perturbed_image_batch)

                if i < kwargs['iters_no_rand']:
                    batch_input_to_model = perturbed_image_batch
                else:
                    batch_input_to_model = self._transform_batch(perturbed_image_batch)

                batch_embedding = self.model(batch_input_to_model)

                batch_embedding = self._l2_normalize(batch_embedding)

                difference = self._l2_distance(target_embedding, batch_embedding)
                mean_difference = tf.reduce_mean(difference)

                self.losses.append(mean_difference)

            if mean_difference < previous_difference:
                previous_difference = mean_difference
                best_perturbation = perturbed_image_batch

            gradient = tape.gradient(difference, perturbed_image_batch)
            sign_of_gradient = tf.cast(tf.sign(gradient), perturbed_image_batch.dtype)

            # Subtract the gradient because we want to minimize the distance
            perturbed_image_batch = perturbed_image_batch - sign_of_gradient * kwargs['alpha']
            perturbed_image_batch = tf.clip_by_value(perturbed_image_batch, image_batch - epsilon, image_batch + epsilon)
            perturbed_image_batch = tf.clip_by_value(perturbed_image_batch, kwargs['bounds'][0], kwargs['bounds'][1])

        return best_perturbation
