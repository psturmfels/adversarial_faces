from matplotlib import pyplot as plt
import h5py
import seaborn
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import pairwise_distances
import sys
from utils import l2_normalize, prewhiten, read_sampled_identities, plot_recall_vary_decoy, recall_given_dist, plot_topk
from PIL import Image
seaborn.set()

attack_name = "community_naive_random"

kwargs = {
    "attack_types": ["community_naive_random", "community_naive_same", "community_naive_mean", "community_sample_gaussian_model"],
    "path_to_adversarial": "/data/vggface/test_perturbed_sampled/{true}/{attack_type}/{target}/epsilon_{epsilon}.h5",
    "path_to_clean": "/data/vggface/test_preprocessed_sampled/{id}/embeddings.h5",
#     "epsilons": [0.0, 0.02, 0.04, 0.06, 0.08, 0.1],
    "epsilon": 0.1,
    "identities": read_sampled_identities("sampled_identities.txt").keys(),
    "colors": ['#020024', '#aaa5f9', '#00f5fb', '#073899','#136703', '#29d649'],
    "names": ["randomly sampled lookup set photo", "same universal target",  "mean of lookup set", "sampled from gaussian"]
}

plot_recall_vary_decoy(
    k=100,
    mode="recall",
    sample_sizes=np.arange(2, 900, 50),
    **kwargs
)

