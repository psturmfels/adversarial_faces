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
                    os.path.join(VGG_BASE, 'test_preprocessed'),
                    'Top level directory for images')
flags.DEFINE_string('model_path',
                    'keras-facenet/model/facenet_keras.h5',
                    'Path to keras model')

def _read_embeddings(identity):
    """
    Helper function to read h5 dataset files.
    """
    image_file = os.path.join(FLAGS.preprocessed_directory,
                              identity,
                              'embeddings.h5')
    with h5py.File(image_file, 'r') as f:
        return f['embeddings'][:]

def main(argv=None):
    seed = 2104202021
    random.seed(2104202021)
    np.random.seed(2104202021)

    identities = os.listdir(FLAGS.preprocessed_directory)
    positive = []
    negative = []
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
        negative_vectors = []
        seen = set()
        # to get roughly the same number of comparison as the base identity to itself,
        # which is n choose 2, we will compute (n_true - 2)/2 other vectors;
        # then, when we do pairwise comparisons between n and (n-2)/2 vectors,
        # we get exactly n choose 2 ground truth negative vectors
        while len(negative_vectors) < ((n_true - 2) // 2):
            other_id = np.random.choice(other_identities)
            other_embeddings = _read_embeddings(other_id)
            embedding_index = np.random.choice(len(other_embeddings))
            if not ((other_id, embedding_index)) in seen:
                seen.add((other_id, embedding_index))
                negative_vectors.append(other_embeddings[embedding_index])

        negative_distances = pairwise_distances(
                this_id_vectors,
                negative_vectors,
                metric='euclidean',
                n_jobs=4
        )
        negative.extend(negative_distances.flatten())

    thresholds = np.arange(1e-6, 3.0, 0.05)
    tprs = []
    fprs = []
    n_pos = float(len(positive))
    n_neg = float(len(negative))
    for t in thresholds:
        tp = np.sum(positive < t)
        fp = np.sum(negative < t)
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)

    with h5py.File("results/roc_curve_{}.txt".format(FLAGS.preprocessed_directory.split("/")[-1]), "w") as f:
        f.create_dataset('positive', data=positive)
        f.create_dataset('negative', data=negative)
        f.create_dataset('thresholds', data=thresholds)
        f.create_dataset('fprs', data=fprs)
        f.create_dataset('tprs', data=tprs)


if __name__ == '__main__':
    app.run(main)



