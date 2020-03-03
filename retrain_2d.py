import os
import h5py
import tensorflow as tf
import numpy as np

from utils import set_up_environment, prewhiten, l2_normalize
from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_string('image_directory',
                    '/projects/leelab3/image_datasets/vgg_face/test_preprocessed/',
                    'Top level directory for images')
flags.DEFINE_string('visible_devices',
                    '0',
                    'CUDA parameter')
flags.DEFINE_string('model_path',
                    'facenet_keras.h5',
                    'Path to keras model')
flags.DEFINE_string('save_path',
                    'facenet_2d.h5',
                    'Path to save fine-tuned model in')
flags.DEFINE_integer('batch_size',
                     64,
                     'Batch size to use during training')
flags.DEFINE_integer('num_identities',
                     5,
                     'Number of identities to re-train on')
flags.DEFINE_integer('num_steps',
                     1000,
                     'Number of gradient steps to take')
flags.DEFINE_float('learning_rate',
                   0.01,
                   'Learning rate for fine-tuning')
flags.DEFINE_float('alpha',
                   0.4,
                   'Loss function temperature')

def _read_identity(identity):
    """
    Helper function to read h5 dataset files.
    """
    image_file = os.path.join(FLAGS.image_directory,
                              identity,
                              'images.h5')
    with h5py.File(image_file, 'r') as f:
        images = f['images'][:]
    images_whitened = prewhiten(images).astype(np.float32)
    return images_whitened

def train(argv=None):
    set_up_environment(visible_devices=FLAGS.visible_devices)
    base_model = tf.keras.models.load_model(FLAGS.model_path)

    fine_tune_model = tf.keras.models.Sequential()
    fine_tune_model.add(base_model)
    fine_tune_model.add(tf.keras.layers.Dense(2, activation=None))
    fine_tune_model.layers[0].trainable = False

    identities = os.listdir(FLAGS.image_directory)
    training_images = []

    for i in range(FLAGS.num_identities):
        identity = identities[i]
        batch_images = _read_identity(identity)
        training_images.append(batch_images)

    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)

    def loss(model, anchor, same_batch, different_batch):
        anchor_embedding = l2_normalize(model(anchor))
        same_embedding   = l2_normalize(model(same_batch))
        different_embedding = l2_normalize(model(different_batch))

        min_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(anchor_embedding - same_embedding), axis=-1)))
        max_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(anchor_embedding - different_embedding), axis=-1)))
        return min_loss - max_loss + FLAGS.alpha, min_loss, max_loss

    def grad(model, anchor, same_batch, different_batch):
        with tf.GradientTape() as tape:
            loss_value, min_loss, max_loss = loss(model, anchor, same_batch, different_batch)
        return loss_value, min_loss, max_loss, tape.gradient(loss_value, model.trainable_variables)

    for step in range(FLAGS.num_steps):
        anchor_index, different_index = np.random.choice(FLAGS.num_identities, size=2, replace=False)

        selected_image_indices = np.random.choice(len(training_images[anchor_index]),
                                                  size=2 * FLAGS.batch_size,
                                                  replace=False)
        selected_diff_indices = np.random.choice(len(training_images[different_index]),
                                                 size=FLAGS.batch_size,
                                                 replace=False)

        anchor_batch = training_images[anchor_index][selected_image_indices[:FLAGS.batch_size]]
        same_batch = training_images[anchor_index][selected_image_indices[FLAGS.batch_size:]]
        different_batch = training_images[different_index][selected_diff_indices]

        loss_value, min_loss, max_loss, grads = grad(fine_tune_model, anchor_batch, same_batch, different_batch)
        optimizer.apply_gradients(zip(grads, fine_tune_model.trainable_variables))

        if step % 50 == 0:
            print('Step {}/{}, loss = {:.6f} ({:.6f}, {:.6f})'.format(step,
                                                                      FLAGS.num_steps,
                                                                      loss_value,
                                                                      min_loss,
                                                                      max_loss))

    tf.keras.models.save_model(fine_tune_model, FLAGS.save_path)

if __name__ == '__main__':
    app.run(train)
