import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import h5py
from tqdm import tqdm
from skimage.util import img_as_ubyte
from sklearn.metrics import pairwise_distances

from utils import prewhiten, l2_normalize

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('output_directory',
                    '/projects/leelab3/image_datasets/vgg_face/test_perturbed/',
                    'Top level directory to output adversarially-modified images')
flags.DEFINE_string('embedding_directory',
                    '/projects/leelab3/image_datasets/vgg_face/test_preprocessed/',
                    'Top level directory for embeddings')
flags.DEFINE_string('attack_type',
                    'self_distance',
                    'One of `self_distance`, `target_image`, `none`')
flags.DEFINE_float('epsilon',
                   0.04,
                   'Maximum perturbation distance for adversarial attacks.')

def _read_identity(identity,
                   top_dir,
                   file_name='images.h5',
                   dataset_name='images',
                   prewhiten=False):
    """
    Helper function to read h5 dataset files.
    """
    dataset_file = os.path.join(top_dir,
                                identity,
                                file_name)
    with h5py.File(dataset_file, 'r') as f:
        data = f[dataset_name][:]
    if prewhiten:
        data = prewhiten(data).astype(np.float32)
    return data

def _top_k_recall(sorted_indices,
                  distances,
                  compare_identities,
                  identity,
                  k=10):
    """
    Helper function to get the number of faces close to
    a given face.
    """
    top_k_identities = compare_identities[sorted_indices[:k]]
    return np.sum(top_k_identities == identity)

def run_attack(argv=None):
    identities = os.listdir(FLAGS.embedding_directory)

    # First we pre-load all of the embeddings into a dictionary.
    print('Preloading embeddings...')
    identity_vector = []
    clean_embeddings_matrix = []
    modified_embeddings_matrix = []
    for identity in tqdm(identities):
        embeddings = _read_identity(identity=identity,
                                    top_dir=FLAGS.embedding_directory,
                                    file_name='embeddings.h5',
                                    dataset_name='embeddings',
                                    prewhiten=False)
        clean_embeddings_matrix.append(embeddings)

        if FLAGS.attack_type == 'none':
            modified_embeddings = embeddings
        else:
            data_directory = os.path.join(FLAGS.output_directory,
                                          identity,
                                          FLAGS.attack_type)
            modified_embeddings = _read_identity(identity=identity,
                                                 top_dir=data_directory,
                                                 file_name='epsilon_{}.h5'.format(FLAGS.epsilon),
                                                 dataset_name='embeddings',
                                                 prewhiten=False)
        modified_embeddings_matrix.append(modified_embeddings)

        for _ in range(len(embeddings)):
            identity_vector.append(identity)

    clean_embeddings_matrix = np.concatenate(clean_embeddings_matrix, axis=0)
    modified_embeddings_matrix = np.concatenate(modified_embeddings_matrix, axis=0)
    identity_vector = np.array(identity_vector)

    performance_dict = {
        'identity': [],
        'image_index': [],
        'recall_count': [],
        'k': [],
        'num_possible': []
    }

    for identity_index, identity in enumerate(identities):
        print('========Running on identity {}, {}/{}========'.format(identity,
                                                                     identity_index,
                                                                     len(identities)))
        clean_embeddings = clean_embeddings_matrix[identity_vector == identity]
        modified_embeddings = modified_embeddings_matrix[identity_vector == identity]

        background_embeddings = clean_embeddings_matrix[identity_vector != identity]
        background_identities = identity_vector[identity_vector != identity]

        modified_identities = np.array([identity] * (len(modified_embeddings) - 1),
                                       dtype=background_identities.dtype)

        # Now, for each image that belongs to this identity, we get the
        # distance between the clean image and all other images:
        # the modified remaining images and the background clean images
        for embedding_index, current_clean_embedding in enumerate(tqdm(clean_embeddings)):
            current_clean_embedding = np.expand_dims(current_clean_embedding, axis=0)

            remaining_modified_embeddings = np.delete(modified_embeddings,
                                                      embedding_index,
                                                      axis=0)
            compare_embeddings = np.concatenate([background_embeddings,
                                                remaining_modified_embeddings],
                                                axis=0)
            compare_identities = np.concatenate([background_identities,
                                                 modified_identities],
                                                 axis=0)

            distances_to_clean_embedding = pairwise_distances(current_clean_embedding,
                                                              compare_embeddings,
                                                              metric='euclidean',
                                                              n_jobs=10)
            distances_to_clean_embedding = np.squeeze(distances_to_clean_embedding)
            sorted_distances_indices = np.argsort(distances_to_clean_embedding)

            # Now that we have the distances between the clean embedding
            # and all other images, we check how many times a match is found
            for k in [1, 10, 100, 1000]:
                recall_count = _top_k_recall(sorted_distances_indices,
                                             distances_to_clean_embedding,
                                             compare_identities,
                                             identity,
                                             k=k)
                performance_dict['identity'].append(identity)
                performance_dict['image_index'].append(embedding_index)
                performance_dict['recall_count'].append(recall_count)
                performance_dict['k'].append(k)
                performance_dict['num_possible'].append(len(clean_embeddings))

    # Finally, we write this data to a massive csv file
    # I anticipate each csv will have nearly 200,000 rows.
    # However, I would rather write the raw data and do aggregation afterwards,
    # just in case we want to plot the data in many different ways.
    os.makedirs('results/{}/'.format(FLAGS.attack_type), exist_ok=True)
    performance_df = pd.DataFrame(performance_dict)
    performance_df.to_csv('results/{}/epsilon_{}.csv'.format(FLAGS.attack_type,
                                                             FLAGS.epsilon),
                          index=False)

if __name__ == '__main__':
    app.run(run_attack)
