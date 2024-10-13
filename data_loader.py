import os
import tensorflow as tf
import imageio

def load_image_paths(image_dir, mask_dir):
    image_list_orig = os.listdir(image_dir)
    image_list = [os.path.join(image_dir, i) for i in image_list_orig]
    mask_list = [os.path.join(mask_dir, i) for i in image_list_orig]
    return image_list, mask_list

def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask

def preprocess(image, mask):
    input_image = tf.image.resize(image, (96, 128), method='nearest')
    input_mask = tf.image.resize(mask, (96, 128), method='nearest')
    return input_image, input_mask

def prepare_dataset(image_list, mask_list, batch_size, buffer_size):
    image_filenames = tf.constant(image_list)
    mask_filenames = tf.constant(mask_list)
    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, mask_filenames))
    
    image_ds = dataset.map(process_path)
    processed_image_ds = image_ds.map(preprocess)
    
    return processed_image_ds.cache().shuffle(buffer_size).batch(batch_size)
