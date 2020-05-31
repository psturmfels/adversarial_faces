import tensorflow as tf
import numpy as np
import h5py
from sklearn.metrics import pairwise_distances
import os
from PIL import Image

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

def identropy_given_dist(
    dist_self,
    dist_negative,
    k,
    identities_negative
):
    self_identity = "selfXXXuniquegibberish"

    all_dist = np.concatenate((dist_self, dist_negative), axis=0)
    all_ids = np.concatenate(([self_identity for x in range(len(dist_self))], identities_negative))

    sorted_indices = np.argsort(all_dist)
    topk_ids = float(len(set(all_ids[sorted_indices][:k])))
    total_ids = float(len(set(all_ids)))
    return topk_ids / total_ids

def numids_given_dist(
    dist_self,
    dist_negative,
    k,
    identities_negative,
    total_ids
):
    self_identity = "selfXXXuniquegibberish"

    all_dist = np.concatenate((dist_self, dist_negative), axis=0)
    all_ids = np.concatenate(([self_identity for x in range(len(dist_self))], identities_negative))

    sorted_indices = np.argsort(all_dist)
    topk_ids = float(len(set(all_ids[sorted_indices][:k])))
    #return np.log2(1.0 / topk_ids + 1.0)
    #return np.power(2.0, 1.0 / topk_ids)
    #return topk_ids
    #total_ids = float(len(set(all_ids)))

    return (total_ids - topk_ids) / total_ids

def recall(
    base_embeddings,
    negative_embeddings,
    k,
    mode="recall",
    target_indices=None,
    neg_identities=None,
    total_ids=0
):
    self_distances = pairwise_distances(
        base_embeddings,
        base_embeddings
    )

    negative_distances = pairwise_distances(
        base_embeddings,
        negative_embeddings
    )


    if func is None:
        raise Exception("Unsupported mode {}".format(mode))

    recall = []
    if not (target_indices is None):
        target_indices = np.int32(np.array(target_indices))

    for indx, dist_self in enumerate(self_distances):
        dist_self = np.delete(dist_self, indx)
        dist_negative = negative_distances[indx]

        if not (target_indices is None):
            # WARNING: assumption here is that target indices refer to the same indices as the
            # base embeddings we provided to this function
            current_indx = np.int32(np.ones(len(target_indices)) * indx)
            dist_negative = dist_negative[target_indices != current_indx]

        try:
            if mode == "identropy":
                r = identropy_given_dist(dist_self, dist_negative, k, neg_identities)
            elif mode == "numids":
                assert total_ids > 0
                r = numids_given_dist(dist_self, dist_negative, k, neg_identities, total_ids)
            else:
                if mode == "recall":
                    func = recall_given_dist
                elif mode == "discovery":
                    func = discovery_given_dist
                r = func(dist_self, dist_negative, k)

        except IndexError as e:
           continue

        recall.append(r)
    return np.mean(recall)

class EmbeddingsProducer:
    '''
    class to produce embedding given format of path to adversarial images
    '''
    def __init__(self, path_to_adversarial, model_path=None):
        self.target_indices_seen = False
        self.path_to_adversarial = path_to_adversarial
        if path_to_adversarial.endswith(".h5"):
            self.raw_images = True # raw images == images from .h5 file
        else:
            self.raw_images = False
            self.model = tf.keras.models.load_model(model_path)

    def _compute_embeddings(self, folder):
        total = len(os.listdir(folder))
        ext = os.path.split(folder)[1]#assuming folder name is same as image extensions
        imgs = [np.array(Image.open(os.path.join(folder, "{}.{}".format(i, ext)))) for i in range(total)]
        imgs = np.array(imgs)
        imgs = prewhiten(imgs)
        return l2_normalize(self.model.predict(imgs))

    def get_embeddings(self, adversarial_target, modified_identity, epsilon):
        embeddings = None
        target_indices = None
        if self.raw_images:
            path_to_h5 = self.path_to_adversarial.format(
                 target=adversarial_target,
                 true=modified_identity,
                 epsilon=epsilon
            )
        else:
            fold = self.path_to_adversarial.format(
                 target=adversarial_target,
                 true=modified_identity,
                 epsilon=epsilon
            )
            path_to_h5 = os.path.join(os.path.split(fold)[0]) + ".h5"

        with h5py.File(path_to_h5, "r") as f:

            if self.raw_images:
                try:
                    embeddings = f["embeddings"][:]
                except KeyError:
                    print("No embeddings found for", end=' ')
                    print(path_to_h5, end=' ')
                    print([k for k in f.keys()])



            if "target_indices" in f.keys():
                self.target_indices_seen = True
                target_indices = f["target_indices"][:]
            elif self.target_indices_seen:
                raise Exception("One file had target indices but others do not; target indices may be inconsistent.")

        if not self.raw_images:
            embeddings = self._compute_embeddings(self.path_to_adversarial.format(
                target=adversarial_target,
                true=modified_identity,
                epsilon=epsilon
            ))

        return embeddings, target_indices


def recall_for_target(
    adversarial_target,
    identities,
    epsilons,
    path_to_adversarial,
    path_to_clean,
    ks,
    mode="recall",
    model_path=None,
    n_sample=-1
):
    query_embeddings = []
    adv = {eps: [] for eps in epsilons}
    adv_target_indices = {eps: [] for eps in epsilons}
    adv_ids = {eps: [] for eps in epsilons}

    ep = EmbeddingsProducer(path_to_adversarial, model_path=model_path)

    with h5py.File(path_to_clean.format(id=adversarial_target), "r") as f:
        query_embeddings.extend(f["embeddings"][:])

    for modified_identity in identities:
        if modified_identity == adversarial_target:
            continue
        for indx, epsilon in enumerate(epsilons):
            if epsilon == 0.0:
                with h5py.File(path_to_clean.format(id=modified_identity), "r") as f:
                    adv[epsilon].extend(f["embeddings"][:])
                    if mode == "identropy" or mode == "numids":
                        adv_ids[epsilon].extend([modified_identity for _ in range(len(f["embeddings"][:]))])
            else:
                adv_eps, adv_ti = ep.get_embeddings(
                    adversarial_target=adversarial_target,
                    modified_identity=modified_identity,
                    epsilon=epsilon
                )
                adv[epsilon].extend(adv_eps)
                if not (adv_ti is None):
                    adv_target_indices[epsilon].extend(adv_ti)

                if mode == "identropy" or mode == "numids":
                    adv_ids[epsilon].extend([modified_identity for _ in range(len(adv_eps))])


    recall_matrix = [[-1.0 for eps in epsilons] for k in ks]
    for kindx, k in enumerate(ks):
        for epsindx, epsilon in enumerate(epsilons):
            advindices = None
            adversarial = adv[epsilon]
            advids = adv_ids[epsilon]
            num_total_ids = len(set(advids))

            if n_sample > 0:
                chosen = np.int32(np.random.choice(len(adv[epsilon]), n_sample))
                adversarial = np.take(np.array(adv[epsilon]), chosen, axis=0)
                if mode == "identropy" or mode == "numids":
                    advids = np.take(np.array(adv_ids[epsilon]), chosen)
                if epsilon in adv_target_indices.keys() and epsilon != 0.0 and len(adv_target_indices[epsilon]) > 0:
                    advindices = np.take(adv_target_indices[epsilon], chosen, axis=0)

            recall_matrix[kindx][epsindx] = recall(query_embeddings, adversarial, k, mode, advindices, advids, num_total_ids)

    return recall_matrix

def plot_recall(
    identities,
    path_to_adversarial,
    path_to_clean,
    epsilons,
    ks,
    colors,
    mode,
    attack_name,
    metric_name,
    model_path=None,
    sample_n=-1,
    attack_plot_name=None,
    save_base=None,
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
            mode,
            model_path,
            sample_n
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

    ax.set_ylabel("Mean {}".format(metric_name))
    ax.set_xlabel("Epsilon (Perturbation Magnitude)")
    ax.set_title("{} {}".format(
        mode,
        attack_name
    ) if attack_plot_name is None else attack_plot_name)
    ax.set_ylim([-0.1, np.max(recall_for_targets) + 0.1])
    ax.legend()
    if not (save_base is None):
        fig_dir = os.path.join(
                save_base,
                "{}_{}.png".format(mode, attack_name)
        )
        fig = plt.gcf()
        fig.savefig(fig_dir)
    else:
        plt.show()

def plot_recall_vary_decoy(
    identities,
    path_to_adversarial,
    path_to_clean,
    epsilon,
    k,
    colors,
    mode,
    metric_name,
    attack_types,
    sample_sizes,
    clean_lookup_size,
    save_base,
    model_path=None,
    names=None,
):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(nrows=1, ncols=1)
    for atindx, attack_type in enumerate(attack_types):
        recall_for_targets = np.ones((len(sample_sizes), len(identities))) * (-1.0)
        for sample_indx, sample_n in enumerate(sample_sizes):
            for indx, identity in enumerate(identities):
                advpath = path_to_adversarial.format(
                        attack_type=attack_type,
                        target='{target}',
                        true='{true}',
                        epsilon='{epsilon}'
                )
                recall_for_targets[sample_indx, indx] = recall_for_target(
                    adversarial_target=identity,
                    identities=identities,
                    epsilons=[epsilon],
                    path_to_adversarial=advpath,
                    path_to_clean=path_to_clean,
                    ks=[k],
                    mode=mode,
                    model_path=model_path,
                    n_sample=sample_n
                )[0][0]

        recall_for_targets = np.mean(recall_for_targets, axis=1)

        ax.plot(
            [float(x) / float(clean_lookup_size) for x in sample_sizes],
            recall_for_targets,
            color=colors[atindx],
            label=attack_type if names is None else names[atindx],
        )

    #ax.set_ylabel("Mean {}".format(mode))
    #ax.set_xlabel("Proportion of decoys relative to clean lookup")
    #ax.set_title("{} at {} with epsilon={}".format(mode, k, epsilon))
    #ax.set_ylim([-0.1, np.max(recall_for_targets) + 0.1])
    #ax.legend()
    #plt.show()
    ax.set_ylabel("Mean {}".format(metric_name))
    ax.set_xlabel("Proportion of decoys relative to clean lookup")
    ax.set_title("{} vs decoys\n for epsilon={} and k = {}".format(metric_name, epsilon, k))
    ax.set_ylim([-0.1, 1.0 + 0.1])
    ax.legend()
    if not (save_base is None):
        fig_dir = os.path.join(
                save_base,
                "{}_vs_decoy_set_size_{}_{}.png".format(mode, epsilon, k)
        )
        fig = plt.gcf()
        fig.savefig(fig_dir)
    else:
        plt.show()


def plot_topk(
    identities,
    adversarial_target,
    epsilon,
    k,
    path_to_adversarial,
    path_to_clean,
    mode,
    model_path=None
):
    from matplotlib import pyplot as plt
    query_embeddings = []
    epsilons = [epsilon]
    adv = {eps: [] for eps in epsilons}

    with h5py.File(path_to_clean.format(id=adversarial_target), "r") as f:
        query_embeddings.extend(f["embeddings"][:])

    ep = EmbeddingsProducer(path_to_adversarial, model_path=model_path)

    for modified_identity in identities:
        if modified_identity == adversarial_target:
            continue
        if epsilon == 0.0:
            with h5py.File(path_to_clean.format(id=modified_identity), "r") as f:
                adv[epsilon].extend(f["embeddings"][:])
        else:
            adv_eps, adv_ti = ep.get_embeddings(
                    adversarial_target=adversarial_target,
                    modified_identity=modified_identity,
                    epsilon=epsilon
                )
            print("Shape of computed embeddings", adv_eps.shape)
            adv[epsilon].extend(adv_eps)


    self_distances = pairwise_distances(query_embeddings, query_embeddings)
    negative_distances = pairwise_distances(query_embeddings, adv[epsilon])
    print("shape of negative distances", negative_distances.shape)

    n_clean = len(query_embeddings)
    n_rows = int(np.sqrt(n_clean))
    fig, ax = plt.subplots(ncols=n_rows, nrows=n_rows+1, figsize=(30, 30))
    for indx in range(n_clean):
        dist_self = np.sort(np.delete(self_distances[indx], indx))[:k]
        dist_neg = np.sort(negative_distances[indx])[:k]
        print("shape of dist neg", dist_neg.shape)
        print("min max of dist neg", np.min(dist_neg), np.max(dist_neg))
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
