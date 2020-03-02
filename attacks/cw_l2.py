"""
Implements the L2 version of the Carlini-Wagner attack.
Ported into TensorFlow 2.0 from:
https://github.com/carlini/nn_robust_attacks/blob/master/li_attack.py
and
https://github.com/bethgelab/foolbox/blob/3999d4334969b7d3debdf846f3f0965eb9032013/foolbox/attacks/carlini_wagner.py
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from .attack import Attacker

class CWAttacker(Attacker):
    def _get_default_kwargs(self, kwargs):
        """
        Defines some default hyper-parameter values.

        Args:
            kwargs: A parameter dictionary.
        """
        if 'max_iterations' not in kwargs:
            kwargs['max_iterations'] = 1000
        if 'initial_const' not in kwargs:
            kwargs['initial_const'] = 1e-5
        if 'learning_rate' not in kwargs:
            kwargs['learning_rate'] = 5e-3
        if 'largest_const' not in kwargs:
            kwargs['largest_const'] = 2e+1
        if 'const_factor' not in kwargs:
            kwargs['const_factor'] = 2.0
        if 'success_distance' not in kwargs:
            kwargs['success_distance'] = 1.242
        if 'initial_perturbation_dist' not in kwargs:
            kwargs['initial_perturbation_dist'] = 0.025
        if 'verbose' not in kwargs:
            kwargs['verbose'] = False
        if 'attack_criteria' not in kwargs:
            kwargs['attack_criteria'] = 'all'
            # One of all, any or mean
        return kwargs

    def _to_attack_space(self, x, bounds):
        """
        Converts a tensor to hyperbolic tangent space.

        Args:
            x: A tensor
            bounds: Minimum and maximum values
        """
        min_, max_ = bounds
        a = (min_ + max_) / 2
        b = (max_ - min_) / 2
        x = (x - a) / b  # map from [min_, max_] to [-1, +1]
        x = x * 0.999999  # from [-1, +1] to approx. (-1, +1)
        x = tf.atanh(x)  # from (-1, +1) to (-inf, +inf)
        return x

    def _to_model_space(self, x, bounds):
        """
        Converts a tensor back into image space.

        Args:
            x: A tensor
            bounds: Minimum and maximum values
        """
        min_, max_ = bounds
        x = tf.tanh(x) # from (-inf, +inf) to (-1, +1)
        a = (min_ + max_) / 2
        b = (max_ - min_) / 2
        x = x * b + a  # map from (-1, +1) to (min_, max_)
        return x

    def self_distance_attack(self, image_batch, epsilon=0.025, **kwargs):
        """
        Attacks a batch of images using the Carlini Wagner attack and
        the self-distance strategy.

        Args:
            image_batch: A batch of images.
            epsilon: Amount of initial perturbation.
            kwargs: Varies depending on attack.
        """
        kwargs = self._get_default_kwargs(kwargs)
        bounds = tf.reduce_min(image_batch), tf.reduce_max(image_batch)

        # Initialize the perturbed example
        noise = self._generate_noise(epsilon, image_batch)
        initial_w_value = tf.clip_by_value(noise + image_batch, bounds[0], bounds[1])
        initial_w_value = self._to_attack_space(initial_w_value, bounds)
        perturbation_w = tf.Variable(initial_w_value)

        original_embedding = self.model(image_batch)
        original_embedding = self._l2_normalize(original_embedding)

        optimizer = tf.keras.optimizers.Adam(learning_rate=kwargs['learning_rate'])

        current_c = kwargs['initial_const']
        while current_c <= kwargs['largest_const']:
            perturbation_w.assign(initial_w_value)
            def loss():
                x_plus_delta = self._to_model_space(perturbation_w, bounds)
                delta = x_plus_delta - image_batch

                perturbed_embedding = self.model(x_plus_delta)
                perturbed_embedding = self._l2_normalize(perturbed_embedding)

                # Negative sign because we want to maximimize the distance
                model_loss = -self._l2_distance(original_embedding, perturbed_embedding)
                norm_loss  = self._l2_norm(delta, axis=(1, 2, 3))
                return current_c * model_loss + norm_loss

            iterable = range(kwargs['max_iterations'])
            if kwargs['verbose']:
                print('Trying attack with c = {:.4f}'.format(current_c))
                iterable = tqdm(iterable)

            for _ in iterable:
                optimizer.minimize(loss, [perturbation_w])

                # Now we check if we have succeeded in our attack
                x_plus_delta = self._to_model_space(perturbation_w, bounds)
                perturbed_embedding = self.model(x_plus_delta)
                perturbed_embedding = self._l2_normalize(perturbed_embedding)
                model_loss = self._l2_distance(original_embedding, perturbed_embedding)

                if kwargs['attack_criteria'] == 'all':
                    succeeded = tf.reduce_all(model_loss > kwargs['success_distance'])
                elif kwargs['attack_criteria'] == 'any':
                    succeeded = tf.reduce_any(model_loss > kwargs['success_distance'])
                elif kwargs['attack_criteria'] == 'mean':
                    succeeded = tf.reduce_mean(model_loss) > kwargs['success_distance']

                if succeeded:
                    if kwargs['verbose']:
                        print('Attack succeeded with c = {:.4f}'.format(current_c))
                    return x_plus_delta

            # If we made it here, we have to increase the constant and try again
            current_c = current_c * 2.0

        if kwargs['verbose']:
            print('Attack failed. Returning None.')
        # Return None in the case of failure
        return None
