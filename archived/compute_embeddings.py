import os
import cv2
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from skimage.transform import resize

import data
import utils

def write_embeddings(proc_dir='/projects/leelab3/image_datasets/lfw/processed/',
                     embedding_dir='/projects/leelab3/image_datasets/lfw/embeddings/'):
    subdirs = os.listdir(proc_dir)
    model = tf.keras.models.load_model('facenet_keras.h5')

    images = []
    directories = []
    batch_size = 64

    for j, subdir in enumerate(tqdm(subdirs)):
        abs_subdir = os.path.join(proc_dir, subdir)
        face_paths = os.listdir(abs_subdir)

        os.makedirs(os.path.join(embedding_dir, subdir), exist_ok=True)

        for face_path in face_paths:
            abs_face_path = os.path.join(abs_subdir, face_path)
            image = data.preprocess(np.load(abs_face_path)).astype(np.float32)
            images.append(image)

            directories.append(os.path.join(subdir, face_path))

            if len(images) >= batch_size or j == len(subdirs) - 1:
                batch_images = np.stack(images, axis=0)
                batch_embeddings = model(batch_images)
                normalized_embeddings = data.normalize_embedding(batch_embeddings)

                for i, directory in enumerate(directories):
                    full_embedding_dir = os.path.join(embedding_dir, directory)
                    np.save(full_embedding_dir, normalized_embeddings[i])

                images = []
                directories = []

if __name__ == '__main__':
    utils.set_up_environment(visible_devices='0')
    write_embeddings()