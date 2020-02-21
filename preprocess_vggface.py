import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from skimage.transform import resize
from skimage.util import img_as_ubyte

from PIL import Image

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('image_directory',
                    '/projects/leelab3/image_datasets/vgg_face/test/',
                    'Top level directory for images')
flags.DEFINE_string('output_directory',
                    '/projects/leelab3/image_datasets/vgg_face/test_preprocessed/',
                    'Directory to output processed images')
flags.DEFINE_string('bbox_file',
                    '/projects/leelab3/image_datasets/vgg_face/bb_landmark/loose_bb_test.csv',
                    'CSV file containing bounding boxes for each desired image')
flags.DEFINE_integer('resize_dimension',
                     160,
                     'Dimension to resize all images to')

flags.DEFINE_boolean('process_train', False, 'Turn on this flag to process the training set')

def preprocess(argv=None):
    bbox_df = pd.read_csv(FLAGS.bbox_file)

    current_identity = bbox_df.loc[0, 'NAME_ID'].split('/')[0]
    image_batch = []

    for index, row in tqdm(bbox_df.iterrows()):
        name_id, x, y, w, h = row
        x = max(x, 0)
        y = max(y, 0)

        image_path = os.path.join(FLAGS.image_directory, name_id + '.jpg')
        image = Image.open(image_path)
        image = np.array(image)

        cropped_image = image[x:x + w, y:y+h]
        resized_image = resize(cropped_image, (FLAGS.resize_dimension, FLAGS.resize_dimension, 3))
        ubyte_image   = img_as_ubyte(resized_image)
        image_batch.append(ubyte_image)

        if index == len(bbox_df) - 1 or \
           current_identity != bbox_df.loc[index + 1, 'NAME_ID'].split('/')[0]:

            image_batch = np.stack(image_batch, axis=0)
            os.makedirs(os.path.join(FLAGS.output_directory,
                                     current_identity), exist_ok=True)

            file_path = os.path.join(FLAGS.output_directory, current_identity, 'images.h5')
            with h5py.File(file_path, 'w') as dataset_file:
                dataset_file.create_dataset('images', data=image_batch)

            image_batch = []

if __name__ == '__main__':
    app.run(preprocess)
