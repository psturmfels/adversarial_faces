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

VGG_BASE = '/data/vggface'

FLAGS = flags.FLAGS

flags.DEFINE_string('output_directory',
                    os.path.join(VGG_BASE, 'test_perturbed'),
                    'Top level directory to output adversarially-modified images')
flags.DEFINE_string('embedding_directory',
                    os.path.join(VGG_BASE, 'test_preprocessed'),
                    'Top level directory for embeddings')
flags.DEFINE_string('attack_type',
                    'random_target',
                    'One of `self_distance`, `target_image`, `none`')
flags.DEFINE_boolean('modify_dataset',
                     False,
                     'Set to True to modify the dataset as well as the identity')
flags.DEFINE_boolean('modify_image',
                     False,
                     'Set to True to modify the target image')
flags.DEFINE_boolean('modify_identity',
                     False,
                     'Set to True to modify the remaining images in from the identity')
flags.DEFINE_float('subsample_rate',
                   1.0,
                   """Fraction of instances of the identity
                   to modify. A floating point number between 0 and 1.
                   """)
flags.DEFINE_float('epsilon',
                   0.04,
                   'Maximum perturbation distance for adversarial attacks.')
flags.DEFINE_integer('num_matrix_jobs',
                     8,
                     'Number of jobs to use during distance computation.')

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
            file_name = os.path.join(FLAGS.attack_type,
                                     'epsilon_{}.h5'.format(FLAGS.epsilon))
            modified_embeddings = _read_identity(identity=identity,
                                                 top_dir=FLAGS.output_directory,
                                                 file_name=file_name,
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

        if FLAGS.modify_dataset:
            background_embeddings = modified_embeddings_matrix[identity_vector != identity]
        else:
            background_embeddings = clean_embeddings_matrix[identity_vector != identity]

        background_identities = identity_vector[identity_vector != identity]
        modified_identities = np.array([identity] * (len(modified_embeddings) - 1),
                                       dtype=background_identities.dtype)

        # Now, for each image that belongs to this identity, we get the
        # distance between the clean image and all other images:
        # the modified remaining images and the background clean images
        if FLAGS.modify_image:
            image_iterable = enumerate(tqdm(modified_embeddings))
        else:
            image_iterable = enumerate(tqdm(clean_embeddings))

        for embedding_index, current_image_embedding in image_iterable:
            current_image_embedding = np.expand_dims(current_image_embedding, axis=0)

            if FLAGS.modify_identity:
                remaining_identity_embeddings = np.delete(modified_embeddings,
                                                          embedding_index,
                                                          axis=0)


                if FLAGS.subsample_rate < 1.0:
                    remaining_id_embeddings_clean = np.delete(clean_embeddings,
                                                              embedding_index,
                                                              axis=0)
                    total_identity_samples = remaining_identity_embeddings.shape[0]
                    num_to_sample = int(FLAGS.subsample_rate * total_identity_samples)

                    modify_sample_indices = np.random.choice(total_identity_samples,
                                                             size=num_to_sample,
                                                             replace=False)
                    modify_sample_mask = np.full(shape=(total_identity_samples,),
                                                 fill_value=False)
                    modify_sample_mask[modify_sample_indices] = True
                    clean_sample_mask = ~modify_sample_mask

                    sampled_modified = remaining_identity_embeddings[modify_sample_mask]
                    sampled_clean = remaining_id_embeddings_clean[clean_sample_mask]

                    remaining_identity_embeddings = np.concatenate([sampled_modified, sampled_clean],
                                                                   axis=0)
            else:
                remaining_identity_embeddings = np.delete(clean_embeddings,
                                                          embedding_index,
                                                          axis=0)

            compare_embeddings = np.concatenate([background_embeddings,
                                                remaining_identity_embeddings],
                                                axis=0)
            compare_identities = np.concatenate([background_identities,
                                                 modified_identities],
                                                 axis=0)

            distances_to_clean_embedding = pairwise_distances(current_image_embedding,
                                                              compare_embeddings,
                                                              metric='euclidean',
                                                              n_jobs=FLAGS.num_matrix_jobs)
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
    output_file = os.path.join("results", FLAGS.attack_type, 'epsilon_{}.h5'.format(FLAGS.epsilon))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    performance_df = pd.DataFrame(performance_dict)
    performance_df.to_csv(output_file,
                          index=False)

if __name__ == '__main__':
    app.run(run_attack)
