import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import h5py
from tqdm import tqdm
from skimage.util import img_as_ubyte
from sklearn.metrics import pairwise_distances

from utils import set_up_environment, prewhiten, l2_normalize
from attacks.pgd import PGDAttacker

from absl import app, flags

VGG_BASE = '/data/vggface'

FLAGS = flags.FLAGS

flags.DEFINE_string('image_directory',
                    os.path.join(VGG_BASE, 'test_preprocessed_sampled'),
                    'Top level directory for images')
flags.DEFINE_string('output_directory',
                    os.path.join(VGG_BASE, 'test_perturbed_sampled'),
                    'Top level directory to output adversarially-modified images')
flags.DEFINE_string('visible_devices',
                    '0',
                    'CUDA parameter')
flags.DEFINE_string('model_path',
                    'keras-facenet/model/facenet_keras.h5',
                    'Path to keras model')
flags.DEFINE_string('attack_type',
                    'community_naive_random',
                    'One of `self_distance`, `target_image`')
flags.DEFINE_float('epsilon',
                   0.04,
                   'Maximum perturbation distance for adversarial attacks.')
flags.DEFINE_integer('batch_size',
                     128,
                     'Batch size to use for gradient-based adversarial attacks')

def _read_identity(identity):
    """
    Helper function to read h5 dataset files.
    """
    image_file = os.path.join(FLAGS.image_directory,
                              identity,
                              'images.h5')
    with h5py.File(image_file, 'r') as f:
        images = f['images'][:]
    images_whitened = prewhiten(images).astype(np.float32)
    return images_whitened

def _read_embeddings(identity):
    """
    Helper function to read h5 dataset files.
    """
    embeddings_file = os.path.join(FLAGS.image_directory,
                              identity,
                              'embeddings.h5')
    with h5py.File(embeddings_file, 'r') as f:
        return f['embeddings'][:].astype(np.float32)

def _attack_images(attacker,
                   images_whitened,
                   previous_images_whitened):
    """
    Helper function to apply the attack strategy.

    Args:
        attacker: subclass object of Attacker class
        images_whitened: A batch of images
        previous_images_whitened: A batch of images to use as targets. Only
                                  used if attack_type == 'target_image'
    """
    modified_images = []
    for i in tqdm(range(0, len(images_whitened), FLAGS.batch_size)):
        num_to_sample = min(FLAGS.batch_size, len(images_whitened) - i)
        batch_images = images_whitened[i:i + num_to_sample]

        if FLAGS.attack_type == 'self_distance':
            modified_batch = attacker.self_distance_attack(batch_images,
                                                           epsilon=FLAGS.epsilon,
                                                           verbose=False)
        elif FLAGS.attack_type == 'target_image':
            target_indices = np.random.choice(len(previous_images_whitened), size=num_to_sample)
            target_images = previous_images_whitened[target_indices]

            modified_batch = attacker.target_image_attack(batch_images,
                                                          target_images,
                                                          epsilon=FLAGS.epsilon,
                                                          verbose=False)
        elif FLAGS.attack_type == 'random_target':
            target_vectors = np.random.uniform(low=-100.0, high=100.0, size=(num_to_sample, 128))
            modified_batch = attacker.target_vector_attack(batch_images,
                                                          target_vectors,
                                                          normalize_target_embedding=True,
                                                          epsilon=FLAGS.epsilon,
                                                          verbose=False)

        elif FLAGS.attack_type == 'none':
            modified_batch = batch_images
        else:
            raise ValueError('Unrecognized value `{}`'.format(FLAGS.attack_type) + \
                             ' for parameter attack_type')
        modified_images.append(modified_batch)

    modified_images = tf.concat(modified_images, axis=0)
    return modified_images

def run_attack():
    set_up_environment(visible_devices=FLAGS.visible_devices)
    model = tf.keras.models.load_model(FLAGS.model_path)
    attacker = PGDAttacker(model)
    identities = os.listdir(FLAGS.image_directory)

    previous_images_whitened = _read_identity(identities[1])

    for identity_index, identity in enumerate(identities):
        print('========Running on identity {}, {}/{}========'.format(identity,
                                                                     identity_index,
                                                                     len(identities)))
        # Here we have to read in the original images and pre-process them
        images_whitened = _read_identity(identity)
        modified_images = _attack_images(attacker,
                                         images_whitened,
                                         previous_images_whitened)
        modified_embeddings = model.predict(modified_images,
                                            batch_size=FLAGS.batch_size)
        modified_embeddings = l2_normalize(modified_embeddings)

        # Now we write the adversarially-modified images
        # and their corresponding embeddings to a series of h5 datasets.
        data_directory = os.path.join(FLAGS.output_directory,
                                 identity,
                                 FLAGS.attack_type)
        os.makedirs(data_directory, exist_ok=True)
        data_path = os.path.join(data_directory, 'epsilon_{}.h5'.format(FLAGS.epsilon))

        with h5py.File(data_path, 'w') as dataset_file:
            dataset_file.create_dataset('embeddings', data=modified_embeddings)
            dataset_file.create_dataset('images', data=modified_images)

        previous_images_whitened = images_whitened

def run_attack_community():
    set_up_environment(visible_devices=FLAGS.visible_devices)
    model = tf.keras.models.load_model(FLAGS.model_path)
    attacker = PGDAttacker(model)
    identities = os.listdir(FLAGS.image_directory)

    for identity_index, identity in enumerate(identities):
        print('========Running on identity {}, {}/{}========'.format(identity,
                                                                     identity_index,
                                                                     len(identities)))
        images_whitened = _read_identity(identity)

        for target_identity in tqdm(list(set(identities) - set([identity]))):
            target_vectors = _read_embeddings(target_identity)

            if FLAGS.attack_type == "community_naive_same":
                targets = [target_vectors[0] for _ in range(len(images_whitened))]
                del target_vectors
            elif FLAGS.attack_type == "community_naive_random":
                targets = target_vectors[np.random.choice(len(images_whitened), size=len(images_whitened), replace=True)]
                del target_vectors
            else:
                raise Exception("Attack type {} not supported".format(FLAGS.attack_type))

            # do attack here
            # notice that this will need to get modified to process in a batched fashion
            # when we use the full dataset
            # it is also not ideal with the currently subsampled dataset because the batch size is
            # "hard-coded" to be the number of images of the identity that we are modifying
            modified_images = attacker.target_vector_attack(
                    images_whitened,
                    targets,
                    normalize_target_embedding=True,
                    epsilon=FLAGS.epsilon,
                    verbose=False
            )

            modified_embeddings = model.predict(modified_images,
                                                batch_size=FLAGS.batch_size)
            modified_embeddings = l2_normalize(modified_embeddings)

            # Now we write the adversarially-modified images
            # and their corresponding embeddings to a series of h5 datasets.
            data_directory = os.path.join(
                    FLAGS.output_directory,
                    identity,
                    FLAGS.attack_type,
                    target_identity
            )
            os.makedirs(data_directory, exist_ok=True)
            data_path = os.path.join(data_directory, 'epsilon_{}.h5'.format(FLAGS.epsilon))

            with h5py.File(data_path, 'w') as dataset_file:
                dataset_file.create_dataset('embeddings', data=modified_embeddings)
                dataset_file.create_dataset('images', data=modified_images)

def main(argv=None):
    if "community" in FLAGS.attack_type:
        run_attack_community()
    else:
        run_attack()

if __name__ == '__main__':
    app.run(main)
