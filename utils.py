import tensorflow as tf
import numpy as np
import h5py
from sklearn.metrics import pairwise_distances
import os

def set_up_environment(mem_frac=None, visible_devices=None, min_log_level='3'):
    """
    A helper function to set up a tensorflow environment.

    Args:
        mem_frac: Fraction of memory to limit the gpu to. If set to None,
                  turns on memory growth instead.
        visible_devices: A string containing a comma-separated list of
                         integers designating the gpus to run on.
        min_log_level: One of 0, 1, 2, or 3.
    """
    if visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(min_log_level)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if mem_frac is not None:
                    memory_limit = int(10000 * mem_frac)
                    config = [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit)]
                    tf.config.experimental.set_virtual_device_configuration(gpu, config)
                else:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            print(error)

def read_sampled_identities(filepath):
    '''
    A helper function to read the sampled identities file.
    Args:
        filepath: the path to the sampled_identities file generated by sample_identities.py
    Returns:
        a dict mapping identities to the images that were picked for each (all strings)
    '''
    id2im = {}
    with open(filepath, 'r') as f:
        num_id, num_im = f.readline().strip("\n").split(" ")
        num_id, num_im = int(num_id), int(num_im)
        for i in range(num_id):
           identity = f.readline().strip("\n")
           id2im[identity] = [f.readline().strip("\n") for _ in range(num_im)]
    return id2im

def prewhiten(x):
    """
    A helper function to whiten an image, or a batch of images.

    Args:
        x: An image or batch of images.
    """
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def maximum_center_crop(x):
    """
    A helper function to crop an image to the maximum center crop.

    Args:
        x: An image.
    """
    minimum_dimension = min(x.shape[0], x.shape[1])
    extension = int(minimum_dimension / 2)
    center = (int(x.shape[0] / 2), int(x.shape[1] / 2))

    x = x[center[0] - extension:center[0] + extension,
          center[1] - extension:center[1] + extension]
    return x

def l2_normalize(x, axis=-1, epsilon=1e-10):
    """
    Normalizes an embedding to have unit length in the l2 metric.

    Args:
        x: A batch of numpy embeddings
    """
    output = x / np.sqrt(np.maximum(np.sum(np.square(x),
                                           axis=axis,
                                           keepdims=True),
                                    epsilon))
    return output

def recall_given_dist(
    dist_self,
    dist_negative,
    k
):
    dist_self = np.sort(dist_self)
    dist_negative = np.sort(dist_negative)
    i = 0.0
    i_self = 0
    i_neg = 0
    recall_count = 0.0
    while i < k:
        if dist_self[i_self] < dist_negative[i_neg]:
            recall_count += 1.0
            i_self += 1
        else:
            i_neg += 1

        i += 1.0

        if i_self >= len(dist_self):
            break

        if i_neg >= len(dist_negative):
            total_true = min(float(len(dist_self)), k - i)
            remaining_true = max(total_true - i_self, 0.0)
            recall_count += remaining_true
            break
    final = recall_count / min(len(dist_self), k)

    return final

def discovery_given_dist(
    dist_self,
    dist_negative,
    k
):
    dist_self = np.sort(dist_self)
    dist_negative = np.sort(dist_negative)
    i_neg = 0
    for i in range(k):
        if i_neg < len(dist_negative):
            if dist_self[0] < dist_negative[i_neg]:
                return 1.0
            i_neg += 1
        elif i < k:
            return 1.0

    return 0.0

def recall(
    base_embeddings,
    negative_embeddings,
    k,
    mode="recall"
):
    self_distances = pairwise_distances(
        base_embeddings,
        base_embeddings
    )

    negative_distances = pairwise_distances(
        base_embeddings,
        negative_embeddings
    )

    func = recall_given_dist if mode == "recall" else discovery_given_dist if mode == "discovery" else None
    if func is None:
        raise Exception("Unsupported mode {}".format(mode))

    recall = []
    for indx, dist_self in enumerate(self_distances):
        dist_self = np.delete(dist_self, indx)
        dist_negative = negative_distances[indx]
        r = func(dist_self, dist_negative, k)
        recall.append(r)
    return np.mean(recall)

def recall_for_target(
    adversarial_target,
    identities,
    epsilons,
    path_to_adversarial,
    path_to_clean,
    ks,
    mode="recall"
):
    query_embeddings = []
    adv = {eps: [] for eps in epsilons}
    with h5py.File(path_to_clean.format(id=adversarial_target), "r") as f:
        query_embeddings.extend(f["embeddings"][:])

    for modified_identity in identities:
        if modified_identity == adversarial_target:
            continue
        for indx, epsilon in enumerate(epsilons):
            if epsilon == 0.0:
                with h5py.File(path_to_clean.format(id=modified_identity), "r") as f:
                    adv[epsilon].extend(f["embeddings"][:])
            else:
                with h5py.File(path_to_adversarial.format(
                    target=adversarial_target,
                    true=modified_identity,
                    epsilon=epsilon
                ), "r") as f:
                    adv[epsilon].extend(f["embeddings"][:])

    return [[recall(query_embeddings, adv[epsilon], k, mode) for epsilon in epsilons] for k in ks]

def plot_recall(
    identities,
    path_to_adversarial,
    path_to_clean,
    epsilons,
    ks,
    colors,
    mode,
    attack_name
):
    from matplotlib import pyplot as plt
    recall_for_targets = np.ones((len(identities), len(ks), len(epsilons))) * (-1.0)

    for indx, identity in enumerate(identities):
        recall_for_targets[indx, :, :] = recall_for_target(
            identity,
            identities,
            epsilons,
            path_to_adversarial,
            path_to_clean,
            ks,
            mode
        )

    recall_for_targets = np.mean(recall_for_targets, axis=0)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    for indx, k in enumerate(ks):
        ax.plot(
            list(epsilons),
            recall_for_targets[indx],
            label="k={}".format(k),
            color=colors[indx],
        )

    ax.set_ylabel("Mean {}".format(mode))
    ax.set_xlabel("Epsilon (Perturbation Amount)")
    ax.set_title("{} from top hits {}".format(mode, attack_name))
    ax.set_ylim([-0.1, 1.1])
    ax.legend()
    plt.show()

def plot_topk(
    identities,
    adversarial_target,
    epsilon,
    k,
    path_to_adversarial,
    path_to_clean,
    mode
):
    from matplotlib import pyplot as plt
    query_embeddings = []
    epsilons = [epsilon]
    adv = {eps: [] for eps in epsilons}

    with h5py.File(path_to_clean.format(id=adversarial_target), "r") as f:
        query_embeddings.extend(f["embeddings"][:])

    for modified_identity in identities:
        if modified_identity == adversarial_target:
            continue
        if epsilon == 0.0:
            with h5py.File(path_to_clean.format(id=modified_identity), "r") as f:
                adv[epsilon].extend(f["embeddings"][:])
        else:
            with h5py.File(path_to_adversarial.format(
                    target=adversarial_target,
                    true=modified_identity,
                    epsilon=epsilon
                ), "r") as f:
                    adv[epsilon].extend(f["embeddings"][:])

    self_distances = pairwise_distances(query_embeddings, query_embeddings)
    negative_distances = pairwise_distances(query_embeddings, adv[epsilon])

    n_clean = len(query_embeddings)
    n_rows = int(np.sqrt(n_clean))
    fig, ax = plt.subplots(ncols=n_rows, nrows=n_rows+1, figsize=(30, 30))
    for indx in range(n_clean):
        dist_self = np.sort(np.delete(self_distances[indx], indx))[:k]
        dist_neg = np.sort(negative_distances[indx])[:k]
        row = indx // n_rows
        col = indx % n_rows
        ax[row][col].plot(range(min(k, len(dist_self))), dist_self, 'go')
        ax[row][col].plot(range(min(k, len(dist_neg))), dist_neg, 'ro')

        ax[row][col].set_ylabel("Distance to image".format(indx))
        ax[row][col].set_xlabel("Top nth image")
        ax[row][col].set_title("Image index {} recall@{}={:.2f} discovery@{}={:.1f}".format(
            indx, k, recall_given_dist(dist_self, dist_neg, k), k, discovery_given_dist(dist_self, dist_neg, k)))
        ax[row][col].set_ylim([-0.1, 1.4])
    fig.tight_layout(pad=3.0, rect=[0, 0.03, 1, 0.95])
    fig.suptitle("Closest {} images to each image of {} at epsilon={}".format(
        k, adversarial_target, epsilon
    ), fontsize=36)
    plt.show()
