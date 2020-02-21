import os
import cv2
import numpy as np

from tqdm import tqdm
from imageio import imread
from skimage.transform import resize

def crop_face(image, cascade, margin=20, image_size=160):
    faces = cascade.detectMultiScale(image,
                                     scaleFactor=1.1,
                                     minNeighbors=3)
    if len(faces) == 0:
        crop_size = 200
        w, h = image.shape[0:2]
        cropped_image = image[max(int((w - crop_size) / 2), 0):min(int((w + crop_size) / 2), w),
                              max(int((h - crop_size) / 2), 0):min(int((h + crop_size) / 2), h)]
        aligned_image = resize(cropped_image, (image_size, image_size), mode='reflect')
        return aligned_image
    else:
        (x, y, w, h) = faces[0]
        cropped = image[y - margin // 2:y + h + margin // 2,
                        x - margin // 2:x + w + margin // 2,
                        :]
        aligned = resize(cropped, (image_size, image_size), mode='reflect')
        return aligned

def preprocess(image):
    if image.ndim == 4:
        axis = (1, 2, 3)
        size = image[0].size
    elif image.ndim == 3:
        axis = (0, 1, 2)
        size = image.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(image, axis=axis, keepdims=True)
    std  = np.std(image,  axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    preprocessed_image = (image - mean) / std_adj

    return preprocessed_image

def align_dataset(raw_dir='/projects/leelab3/image_datasets/lfw/raw/',
                  proc_dir='/projects/leelab3/image_datasets/lfw/processed/'):
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    subdirs = os.listdir(raw_dir)
    for subdir in tqdm(subdirs):

        abs_subdir = os.path.join(raw_dir, subdir)
        face_paths = os.listdir(abs_subdir)

        os.makedirs(os.path.join(proc_dir, subdir), exist_ok=True)

        for face_path in face_paths:
            abs_face_path  = os.path.join(abs_subdir, face_path)
            save_face_path = os.path.join(proc_dir, subdir, face_path)

            image = imread(abs_face_path)

            try:
                cropped_image = crop_face(image, cascade)
                np.save(save_face_path, cropped_image)
            except ValueError:
                pass

def rename_dataset(proc_dir='/projects/leelab3/image_datasets/lfw/processed/'):
    subdirs = os.listdir(proc_dir)
    for subdir in tqdm(subdirs):
        abs_subdir = os.path.join(proc_dir, subdir)
        face_paths = os.listdir(abs_subdir)
        for face_path in face_paths:
            abs_face_path = os.path.join(abs_subdir, face_path)
            index = abs_face_path.find('.jpg')
            new_face_path = abs_face_path[:index] + abs_face_path[index + 4:]
            os.rename(abs_face_path, new_face_path)

def get_dataset_dict(proc_dir='/projects/leelab3/image_datasets/lfw/processed/'):
    data_dict = {}

    subdirs = os.listdir(proc_dir)
    for subdir in tqdm(subdirs):
        data_dict[subdir] = []
        abs_subdir = os.path.join(proc_dir, subdir)
        face_paths = os.listdir(abs_subdir)
        for face_path in face_paths:
            abs_face_path = os.path.join(abs_subdir, face_path)

            data_dict[subdir].append(abs_face_path)
    return data_dict

def normalize_embedding(embedding, axis=-1, epsilon=1e-10):
    normalized_embedding = embedding / np.sqrt(np.maximum(np.sum(np.square(embedding),
                                                   axis=axis,
                                                   keepdims=True),
                                            epsilon))
    return normalized_embedding


if __name__ == '__main__':
#     align_dataset()
    rename_dataset()