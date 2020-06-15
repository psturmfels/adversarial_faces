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
#VGG_BASE = '/projects/leelab3/image_datasets/vgg_face/'

FLAGS = flags.FLAGS

flags.DEFINE_string('image_directory',
                    os.path.join(VGG_BASE, 'test_preprocessed'),
                    'Top level directory for images')
flags.DEFINE_string('output_directory',
                    os.path.join(VGG_BASE, 'test_perturbed'),
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
flags.DEFINE_integer('num_lookup',
                     49, # 49 is half of the min number of images for any identity in VGGFace2
                     'The number of photos to use as "lookup photos" for computing targets')
flags.DEFINE_integer('num_decoys',
                     10, # 49 is half of the min number of images for any identity in VGGFace2
                     'The number of photos from the decoy identity to use as decoys for protection')

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

def _read_adv_embeddings(identity, target):
    """
    Helper function to read h5 dataset files.
    """
    embeddings_file = os.path.join(
            FLAGS.output_directory,
            identity,
            FLAGS.attack_type,
            target
    )
    embeddings_file = os.path.join(FLAGS.image_directory,
                              identity,
                              'embeddings.h5')
    with h5py.File(embeddings_file, 'r') as f:
        return f['embeddings'][:].astype(np.float32)

class TargetsGenerator:
    def __init__(self, protected_embeddings):
        self.protected_embeddings = np.array(protected_embeddings)
        self.mean_protected_embeddings = np.mean(self.protected_embeddings, axis=0)
        self.std_protected_embeddings = np.std(self.protected_embeddings, axis=0)
        self.num_protected_embeddings, self.dim_embeddings = self.protected_embeddings.shape
        self.choice_indices = []

    def get_targets(self, num_targets):
        if FLAGS.attack_type == "community_naive_same":
            self.choice_indices = np.tile(0, num_targets)
            return np.tile(self.protected_embeddings[0], (num_targets, 1))

        elif FLAGS.attack_type == "community_naive_random":
            self.choice_indices = np.random.choice(
                    self.num_protected_embeddings,
                    size=num_targets,
                    replace=True)
            return self.protected_embeddings[choice_indices]

        elif FLAGS.attack_type == "community_naive_mean":
            return np.tile(self.mean_protected_embeddings, (num_targets, 1))

        elif FLAGS.attack_type == "community_sample_gaussian_model":
            return np.random.normal(
                    self.mean_protected_embeddings,
                    self.std_protected_embeddings,
                    size=(num_targets, self.dim_embeddings))
        else:
            raise Exception("Attack type {} not supported".format(FLAGS.attack_type))


def _get_targets(target_arrays, num_targets):
    if FLAGS.attack_type == "community_naive_same":
        targets = target_arrays[0]
        targets = np.expand_dims(targets, axis=0)
        targets = np.tile(targets, (num_targets, 1, 1, 1))
    elif FLAGS.attack_type == "community_naive_random":
        choice_indices = np.random.choice(len(target_arrays),
                                          size=num_targets,
                                          replace=True)
        targets = target_arrays[choice_indices]
    elif FLAGS.attack_type == "community_naive_mean":
        mean_target = np.mean(np.array(target_arrays), axis=0)
        targets = np.expand_dims(mean_target, axis=0)
        targets = np.tile(targets, (num_targets, 1, 1, 1))
    elif FLAGS.attack_type == "community_sample_gaussian_model":
        if len(target_arrays.shape) > 2:
            raise ValueError('This attack is only supported when passing in '
                             'embedding layers, not images.')
        mean_target = np.mean(np.array(target_arrays),
                              axis=0)
        std_target = np.std(np.array(target_arrays),
                            axis=0)
        targets = np.random.normal(mean_target,
                                   std_target,
                                   size=(num_targets,
                                         len(mean_target)))
    else:
        raise Exception("Attack type {} not supported".format(FLAGS.attack_type))
    return targets

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

def run_attack_community_global():
    set_up_environment(visible_devices=FLAGS.visible_devices)
    model = tf.keras.models.load_model(FLAGS.model_path)
    attacker = PGDAttacker(model)
    identities = os.listdir(FLAGS.image_directory)

    # Ensure consistency in identity ordering
    identities = list(np.sort(identities))

    # We will send decoy photos to the first 50 identities,
    # We will also hold out the first 100 photos of each target
    # identities as test set photos
    target_identities = identities[:50]
    helper_identities = identities[50:]

    for identity_index, target_identity in enumerate(target_identities):
        print('========Running on identity {}, {}/{}========'.format(target_identity,
                                                                     identity_index,
                                                                     len(identities)))
        target_images = _read_identity(target_identity)
        target_images_heldout = target_images[:100]
        target_images_lookup  = target_images[100:]

        target_embeddings = _read_embeddings(target_identity)
        target_embeddings_heldout = target_embeddings[:100]
        target_embeddings_lookup  = target_embeddings[100:]

        decoy_images = []
        decoy_embeddings = []
        decoy_identities = []
        decoy_indices    = []

        helper_image_queue = []
        for helper_index, helper_identity in enumerate(helper_identities):
            helper_images = _read_identity(helper_identity)
            # We will send the first 10 images of each helper identity
            # to the target identity, giving us a total of 4500 decoy
            # images per target identity, which we can subsample accordingly.
            # At maximum, this is roughly 10x the images in the target's lookup set.
            helper_image_queue.append(helper_images[:10])

            decoy_identities += [helper_index] * 10
            decoy_indices += [list(range(10))]

            if len(helper_image_queue) * 10 >= FLAGS.batch_size:
                batch_helper_images = np.concatenate(helper_image_queue, axis=0)
                batch_target_images = _get_targets(target_arrays=target_images_lookup,
                                                   num_targets=batch_helper_images.shape[0])

                batch_modified_helper = attacker.target_image_attack(image_batch=batch_helper_images,
                                                                     target_batch=batch_target_images,
                                                                     epsilon=FLAGS.epsilon)
                batch_modified_helper_embeddings = model(batch_modified_helper)
                batch_modified_helper_embeddings = l2_normalize(batch_modified_helper_embeddings)

                decoy_images.append(batch_modified_helper)
                decoy_embeddings.append(batch_modified_helper_embeddings)
                helper_image_queue = []

        decoy_images = np.concatenate(decoy_images, axis=0)
        decoy_embeddings = np.concatenate(decoy_embeddings, axis=0)
        decoy_identities = np.array(decoy_identities)
        decoy_indices = np.array(decoy_indices)

        data_directory = os.path.join(
            FLAGS.output_directory,
            target_identity,
            FLAGS.attack_type,
        )
        os.makedirs(data_directory, exist_ok=True)
        data_path = os.path.join(data_directory, 'epsilon_{}.h5'.format(FLAGS.epsilon))

        with h5py.File(data_path, 'w') as dataset_file:
            dataset_file.create_dataset('embeddings', data=decoy_embeddings)
            dataset_file.create_dataset('images', data=decoy_images)
            dataset_file.create_dataset('identities', data=decoy_identities)
            dataset_file.create_dataset('indices', data=decoy_indices)

def run_attack_community():
    set_up_environment(visible_devices=FLAGS.visible_devices)
    model = tf.keras.models.load_model(FLAGS.model_path)
    attacker = PGDAttacker(model)
    identities = os.listdir(FLAGS.image_directory)

    for identity_index, identity in enumerate(tqdm(identities)):
        protected_embeddings = _read_embeddings(identity)[:FLAGS.num_lookup]
        tg = TargetsGenerator(protected_embeddings)

        images_to_modify = []
        modification_targets = []
        orig_id = []

        decoy_images = []
        decoy_targets = []
        decoy_embeddings = []
        decoy_indices = []
        decoy_orig_id = []

        for decoy_identity in list(set(identities) - set([identity])):
            images_to_modify.extend(_read_identity(decoy_identity)[:FLAGS.num_decoys])
            modification_targets.extend(tg.get_targets(FLAGS.num_decoys))
            orig_id.extend(np.tile(decoy_identity, FLAGS.num_decoys))

            if len(images_to_modify) >= FLAGS.batch_size:
                modified_images = attacker.target_vector_attack(
                        images_to_modify,
                        modification_targets,
                        normalize_target_embedding=True,
                        epsilon=FLAGS.epsilon,
                        verbose=False
                )
                decoy_images.extend(modified_images)
                decoy_embeddings.extend(
                        l2_normalize(
                            model.predict(
                                modified_images,
                                batch_size=len(modified_images))))
                decoy_targets.extend(modification_targets)
                decoy_orig_id.extend(orig_id)

                if len(tg.choice_indices) == len(modified_images):
                    decoy_indices.extend(tg.choice_indices)

                images_to_modify = []
                modification_targets = []
                orig_id = []

        data_directory = os.path.join(
                FLAGS.output_directory,
                identity,
                FLAGS.attack_type,
        )
        os.makedirs(data_directory, exist_ok=True)
        data_path = os.path.join(data_directory, 'decoys_epsilon_{}.h5'.format(FLAGS.epsilon))

        with h5py.File(data_path, 'w') as dataset_file:
            dataset_file.create_dataset('num_lookup', data=FLAGS.num_lookup)
            dataset_file.create_dataset('decoy_embeddings', data=decoy_embeddings)
            dataset_file.create_dataset('decoy_images', data=decoy_images)
            dataset_file.create_dataset('decoy_targets', data=decoy_targets)
            dataset_file.create_dataset('decoy_true_identities', data=decoy_orig_id)
            if len(decoy_indices) > 0:
                dataset_file.create_dataset('decoy_target_indices', data=decoy_indices)


def main(argv=None):
    if "community" in FLAGS.attack_type:
        run_attack_community()
    else:
        run_attack()

if __name__ == '__main__':
    app.run(main)
