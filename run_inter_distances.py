import os
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from skimage.util import img_as_ubyte
from sklearn.metrics import pairwise_distances

from utils import set_up_environment, l2_normalize

from absl import app, flags

VGG_BASE = '/projects/leelab3/image_datasets/vgg_face/'

FLAGS = flags.FLAGS

flags.DEFINE_string('embedding_directory',
                    os.path.join(VGG_BASE, 'test_preprocessed'),
                    'Top level directory for embeddings')
flags.DEFINE_integer('num_matrix_jobs',
                     8,
                     'Number of jobs to use during distance computation.')

def _read_identity(identity,
                   top_dir,
                   file_name='embeddings.h5',
                   dataset_name='embeddings'):
    """
    Helper function to read h5 dataset files.
    """
    dataset_file = os.path.join(top_dir,
                                identity,
                                file_name)
    with h5py.File(dataset_file, 'r') as f:
        data = f[dataset_name][:]
    return data

def compute_distances(argv=None):
    identities = os.listdir(FLAGS.embedding_directory)

    print('Preloading embeddings...')
    identity_vector = []
    clean_embeddings_matrix = []
    for identity in tqdm(identities):
        embeddings = _read_identity(identity=identity,
                                    top_dir=FLAGS.embedding_directory,
                                    file_name='embeddings.h5',
                                    dataset_name='embeddings')
        clean_embeddings_matrix.append(embeddings)
        for _ in range(len(embeddings)):
            identity_vector.append(identity)

    clean_embeddings_matrix = np.concatenate(clean_embeddings_matrix, axis=0)
    identity_vector = np.array(identity_vector)

    performance_dict = {
        'identity': [],
        'image_index': [],
        'distance_to_mean': [],
        'number_closer_than_mean': [],
        'min_pairwise_dist': [],
        'mean_pairwise_dist': []
    }

    for identity_index, identity in tqdm(enumerate(identities)):
        clean_embeddings = clean_embeddings_matrix[identity_vector == identity]
        mean_embedding = np.mean(clean_embeddings, axis=0)
        mean_embedding = l2_normalize(mean_embedding)
        mean_embedding = mean_embedding.reshape(1, -1)

        inter_pairwise = pairwise_distances(clean_embeddings,
                                            metric='euclidean',
                                            n_jobs=FLAGS.num_matrix_jobs)
        summed_pairwise = np.sum(inter_pairwise, axis=-1)
        # Subtract 1 here from the denominator because of zero diagonals
        mean_pairwise = summed_pairwise / (summed_pairwise.shape[-1] - 1.0)

        np.fill_diagonal(inter_pairwise, np.inf)
        min_pairwise = np.min(inter_pairwise, axis=-1)

        number_closer = np.sum(inter_pairwise < mean_pairwise[:, np.newaxis], axis=-1)

        dist_to_mean = pairwise_distances(clean_embeddings,
                                          mean_embedding,
                                          metric='euclidean',
                                          n_jobs=FLAGS.num_matrix_jobs)[:, 0]

        for embedding_index in range(clean_embeddings.shape[0]):
            performance_dict['identity'].append(identity)
            performance_dict['image_index'].append(embedding_index)
            performance_dict['distance_to_mean'].append(dist_to_mean[embedding_index])
            performance_dict['min_pairwise_dist'].append(min_pairwise[embedding_index])
            performance_dict['number_closer_than_mean'].append(number_closer[embedding_index])
            performance_dict['mean_pairwise_dist'].append(mean_pairwise[embedding_index])

    output_file = 'results/distances.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    performance_df = pd.DataFrame(performance_dict)
    performance_df.to_csv(output_file,
                          index=False)

if __name__ == '__main__':
    app.run(compute_distances)
