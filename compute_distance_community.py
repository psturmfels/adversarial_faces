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

flags.DEFINE_string('perturbed_directory',
                    os.path.join(VGG_BASE, 'test_perturbed_sampled'),
                    'Top level directory to output adversarially-modified images')
flags.DEFINE_string('clean_directory',
                    os.path.join(VGG_BASE, 'test_preprocessed_sampled'),
                    'Top level directory of the clean dataset')
flags.DEFINE_string('attack_type',
                    'community_naive_same',
                    'One of `self_distance`, `target_image`, `none`')
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
                  compare_identities,
                  identity,
                  k=10):
    """
    Helper function to get the number of faces close to
    a given face.
    """
    top_k_identities = compare_identities[sorted_indices[:k]]
    return np.sum(top_k_identities == identity)

def _adversarial_set_for_identity(
        query_identities,
        current_query_identity
):
    '''
    given preloaded query_identities and current query_id being processed,
    loads and returns the portion of the lookup set consisting of
    the adversarially modified images of other identities that target
    this current query_identity
    '''
    all_embeddings = []
    for other_identity in query_identities:
        if other_identity == current_query_identity:
            continue
        dataset_file = os.path.join(
            FLAGS.perturbed_directory,
            current_query_identity,
            FLAGS.attack_type,
            other_identity,
            "epsilon_{}.h5".format(FLAGS.epsilon)
        )
        with h5py.File(dataset_file, 'r') as f:
            all_embeddings.extend(f["embeddings"][:])
    return np.array(all_embeddings)

def _write_to_csv(performance_dict):
    output_file = os.path.join("results", FLAGS.attack_type, 'epsilon_{}.csv'.format(FLAGS.epsilon))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    performance_df = pd.DataFrame(performance_dict)
    performance_df.to_csv(output_file,
                          index=False)

def compute_distance_global(argv=None):
    identities = os.listdir(FLAGS.clean_directory)

    print('Preloading embeddings...')
    identity_vector = []
    clean_embeddings_matrix = []
    for identity in tqdm(identities):
        embeddings = _read_identity(identity=identity,
                                    top_dir=FLAGS.embedding_directory,
                                    file_name='embeddings.h5',
                                    dataset_name='embeddings',
                                    prewhiten=False)
        clean_embeddings_matrix.append(embeddings)
        identity_vector += [identity] * len(embeddings)

    clean_embeddings_matrix = np.concatenate(clean_embeddings_matrix, axis=0)
    identity_vector = np.array(identity_vector)

    target_identities = identities[:50]
    helper_identities = identities[50:]

    performance_dict = {
        'identity': [],
        'image_index': [],
        'recall_count': [],
        'k': [],
        'num_lookup': [],
        'num_decoy': []
    }
    for identity_index, target_identity in enumerate(target_identities):
        print('========Running on identity {}, {}/{}========'.format(identity,
                                                                     identity_index,
                                                                     len(identities)))
        †arget_embeddings = clean_embeddings_matrix[identity_vector == identity]
        target_embeddings_heldout = target_embeddings[:100]
        target_embeddings_lookup  = target_embeddings[100:]

        # We compare distances against the heldout set from the target
        reference_embeddings = clean_embeddings_matrix[identity_vector != identity]
        reference_identities = identity_vector[identity_vector != identity]

        reference_embeddings = np.concatenate([reference_embeddings,
                                               target_embeddings_lookup],
                                              axis=0)
        reference_identities += [target_identity] * target_embeddings_lookup.shape[0]

        # But first we make sure to add the decoy embeddings into the pool of references
        decoy_file = os.path.join(FLAGS.attack_type, 'epsilon_{}.h5'.format(FLAGS.epsilon))
        decoy_embeddings = _read_identity(identity=target_identity,
                                          top_dir=FLAGS.perturbed_directory,
                                          file_name=decoy_file,
                                          dataset_name='embeddings')

        reference_embeddings = np.concatenate([reference_embeddings,
                                               decoy_embeddings],
                                              axis=0)
        reference_identities += ['decoy'] * decoy_embeddings.shape[0]

        # Bulk compute distance between identity heldout and references
        distances_to_clean_embedding = pairwise_distances(target_embeddings_heldout,
                                                          reference_embeddings,
                                                          metric='euclidean',
                                                          n_jobs=FLAGS.num_matrix_jobs)

        for t_index, target_embedding in enuemrate(target_embeddings_heldout):
            current_distances = distances_to_clean_embedding[t_index]
            sorted_distances_indices = np.argsort(current_distances)

            for num_lookup in [1, 5, 10, 20, 50, 100, len(target_embeddings_lookup)]:
                filter_mask = np.full(reference_identites, True)

                # We simulated the target having num_lookup photos in the lookup set
                target_lookup_indices = np.where(reference_identites == target_identity)[0]
                filter_mask[target_lookup_indices[num_lookup:]] = False

                for num_decoy in [100, 1000, decoy_embeddings.shape[0]]:
                    decoy_indices = np.where(reference_identities == 'decoy')[0]
                    filter_mask[decoy_indices[num_decoy]] = False

                    for k in [1, 10, 100, 1000]:
                        test_distances = current_distances[filter_mask]
                        test_identites = reference_identites[filter_mask]
                        recall_count = _top_k_recall(test_distances,
                                                     test_identites,
                                                     target_identity,
                                                     k=k)
                        performance_dict['identity'].append(target_identity)
                        performance_dict['image_index'].append(embedding_index)
                        performance_dict['recall_count'].append(recall_count)
                        performance_dict['k'].append(k)
                        performance_dict['num_lookup'].append(num_lookup)
                        performance_dict['num_decoy'].append(num_decoy)

    output_file = os.path.join("results", FLAGS.attack_type, 'epsilon_{}.h5'.format(FLAGS.epsilon))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    performance_df = pd.DataFrame(performance_dict)
    performance_df.to_csv(output_file,
                          index=False)

def main(argv=None):
    # we will compute the top k images returned for each clean image for each identity
    # terminology:
    # query_identity/query_image = the current clean id/image we are computing
    #   top k recall for
    # lookup_set = the set we are querying to return matches from;
    #   this includes all images of other identities (perhaps modified) *and*
    #   all images of the query_identity that are not the query_image
    query_identities = os.listdir(FLAGS.clean_directory)

    performance_dict = {
        'identity': [],
        'image_index': [],
        'recall_count': [],
        'k': [],
        'num_possible': []
    }

    for query_identity in query_identities:
        curr_query_embeddings = _read_identity(
            query_identity,
            FLAGS.clean_directory,
            "embeddings.h5",
            "embeddings",
            False
        )

        other_embeddings = _adversarial_set_for_identity(
                query_identities,
                query_identity
        )

        lookup_set = np.concatenate([curr_query_embeddings, other_embeddings], axis=0)
        lookup_set_groundtruth = np.array(np.concatenate([ \
                [True for _ in range(len(curr_query_embeddings))],
                [False for _ in range(len(other_embeddings))]
        ]))

        all_pairwise_distances = pairwise_distances(
                curr_query_embeddings,
                lookup_set,
                metric='euclidean',
                n_jobs=FLAGS.num_matrix_jobs
        )

        for query_embedding_index in range(len(curr_query_embeddings)):
            lookup_distances = all_pairwise_distances[query_embedding_index]

            # note that the all_pairwise_distances matrix includes distances to self
            # to remove them, assume the smallest distance is 0 and belongs to the vector itself
            # in the output of argsort, that should come up first
            # TODO: are there cases where this breaks because of e.g. float precision?
            sort_indices = np.argsort(lookup_distances)[1:]

            # with the largest indices into the lookup set in hand, select only indices
            # that are in the top k from the ground truth of the lookup set
            lookup_match_query = lookup_set_groundtruth[sort_indices]
            for k in [1, 10, 100, 1000]:
                recall_count = np.sum(lookup_match_query[:k])

                performance_dict['identity'].append(query_identity)
                performance_dict['image_index'].append(query_embedding_index)
                performance_dict['recall_count'].append(recall_count)
                performance_dict['k'].append(k)
                performance_dict['num_possible'].append(len(curr_query_embeddings) - 1)

    _write_to_csv(performance_dict)


if __name__ == '__main__':
    app.run(compute_distance_global)
