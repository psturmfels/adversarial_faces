import os
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from utils import set_up_environment, maximum_center_crop, prewhiten, l2_normalize, read_sampled_identities
from absl import app, flags
import random

VGG_BASE = '/data/vggface'

FLAGS = flags.FLAGS
flags.DEFINE_string('preprocessed_directory',
                    os.path.join(VGG_BASE, 'test_preprocessed_sampled'),
                    'Top level directory for images')
flags.DEFINE_string('perturbed_directory',
                    os.path.join(VGG_BASE, 'test_perturbed_sampled'),
                    'Top level directory for images')
flags.DEFINE_string('model_path',
                    'keras-facenet/model/facenet_keras.h5',
                    'Path to keras model')
flags.DEFINE_boolean('community_attack',
                    True,
                    'If True, assumes folder structure is that of community attacks and uses adversarial images targeted to identity as negatives')
flags.DEFINE_string('attack_type',
                    'community_naive_same',
                    'Attack type to use when loading images')
flags.DEFINE_string('strategy',
                    'random_image',
                    'Sampling strategy for negative examples; one of `random_image` or `random_identity`')
#random_image continuously samples any other image from any other identity
#random_identity only samples a random other identity and uses all of its images
flags.DEFINE_float('epsilon',
                   0.02,
                   'Maximum perturbation distance for adversarial attacks.')

def _read_embeddings(identity):
    """
    Helper function to read h5 dataset files.
    """
    image_file = os.path.join(FLAGS.preprocessed_directory,
                              identity,
                              'embeddings.h5')
    with h5py.File(image_file, 'r') as f:
        return f['embeddings'][:]

def _read_adversarial_embeddings(true_identity, target_identity):
    """
    Helper function to read h5 dataset files.
    Params:
        true_identity: the ground truth identity of the image that was modified to be target
        target_identity: the targeted identity
    """
    image_file = os.path.join(FLAGS.perturbed_directory,
                              true_identity,
                              FLAGS.attack_type,
                              target_identity,
                              'epsilon_{}.h5'.format(FLAGS.epsilon))
    with h5py.File(image_file, 'r') as f:
        return f['embeddings'][:]


def main(argv=None):
    seed = 2104202021
    random.seed(2104202021)
    np.random.seed(2104202021)

    identities = os.listdir(FLAGS.preprocessed_directory)
    positive = []
    negative = []

    print("****** Computing pairwise distances *******")
    for identity in tqdm(identities):
        this_id_vectors = _read_embeddings(identity)
        n_true = len(this_id_vectors)
        self_distances = pairwise_distances(
                this_id_vectors,
                this_id_vectors,
                metric='euclidean',
                n_jobs=4
        )
        # all distances below the diagonal compare every pair and exclude 0's to self
        self_dist_indices = np.tril_indices(n=n_true, k=-1)
        # this selection returns a flattened array of distances that are ground truth True
        self_distances = self_distances[self_dist_indices]
        positive.extend(self_distances)

        other_identities = list(set(identities) - set([identity]))

        if FLAGS.strategy == 'random_image':
            negative_vectors = []
            seen = set()
            # to get roughly the same number of comparison as the base identity to itself,
            # which is n choose 2, we will compute (n_true - 2)/2 other vectors;
            # then, when we do pairwise comparisons between n and (n-2)/2 vectors,
            # we get exactly n choose 2 ground truth negative vectors
            while len(negative_vectors) < ((n_true - 2) // 2):
                other_id = np.random.choice(other_identities)

                if FLAGS.community_attack:
                    other_embeddings = _read_adversarial_embeddings(
                            true_identity=other_id,
                            target_identity=identity
                    )
                else:
                    other_embeddings = _read_embeddings(other_id)

                embedding_index = np.random.choice(len(other_embeddings))
                if not ((other_id, embedding_index)) in seen:
                    seen.add((other_id, embedding_index))
                    negative_vectors.append(other_embeddings[embedding_index])
        elif FLAGS.strategy == 'random_identity':
            other_id = np.random.choice(other_identities)
            negative_vectors = _read_embeddings(other_id)
        else:
            raise Exception("Unsupported negative examples sampling strategy in FLAG strategy {}".format(FLAGS.strategy))

        negative_distances = pairwise_distances(
                this_id_vectors,
                negative_vectors,
                metric='euclidean',
                n_jobs=4
        )
        negative.extend(negative_distances.flatten())

    thresholds = np.arange(1e-6, 2.0, 0.1)
    tprs = []
    fprs = []
    n_pos = float(len(positive))
    n_neg = float(len(negative))

    print("**** Computing ROC curve *****")
    for t in tqdm(thresholds):
        tp = np.sum(positive < t)
        fp = np.sum(negative < t)
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)

    if not FLAGS.community_attack:
        if not ("perturbed" in  FLAGS.preprocessed_directory):
            save_file_name = "results/roc_curve_{}_{}.txt".format(FLAGS.preprocessed_directory.split("/")[-1], FLAGS.strategy)
        else:
            save_file_name = "results/roc_curve_{attack_type}_epsilon_{epsilon}_{strategy}.txt".format(
                attack_type=FLAGS.attack_type,
                epsilon=FLAGS.epsilon,
                strategy=FLAGS.strategy
        )
    else:
        save_file_name = "results/roc_curve_{perturbed_dir}_{attack_type}_epsilon_{epsilon}_{strategy}.txt".format(
                perturbed_dir=FLAGS.preprocessed_directory.split("/")[-1],
                attack_type=FLAGS.attack_type,
                epsilon=FLAGS.epsilon,
                strategy=FLAGS.strategy
        )

    with h5py.File(save_file_name, "w") as f:
        f.create_dataset('positive', data=positive)
        f.create_dataset('negative', data=negative)
        f.create_dataset('thresholds', data=thresholds)
        f.create_dataset('fprs', data=fprs)
        f.create_dataset('tprs', data=tprs)


if __name__ == '__main__':
    app.run(main)



